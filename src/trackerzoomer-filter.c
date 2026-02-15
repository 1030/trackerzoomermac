#include <obs-module.h>
#include <graphics/graphics.h>
#include <util/platform.h>

#if !defined(_WIN32)
#include <util/threading.h>
#else
// On Windows we use apriltag's Win32 pthread shim (pthreads_cross).
#include "common/pthreads_cross.h"
#ifndef pthread_mutex_init_value
#define pthread_mutex_init_value(m) pthread_mutex_init((m), NULL)
#endif
#endif

#if defined(HAVE_OBS_FRONTEND_API)
#include <obs/obs-frontend-api.h>
#endif

#include <stdio.h>
#include <math.h>

#include "apriltag.h"
#include "tag16h5.h"

// Render-crop shader (samples sub-rectangle of input texture)
static gs_effect_t *g_trackerzoomer_crop_effect = NULL;
static gs_eparam_t *g_param_mul = NULL;
static gs_eparam_t *g_param_add = NULL;

static const char *k_trackerzoomer_crop_effect_src =
	"uniform float4x4 ViewProj;\n"
	"uniform texture2d image;\n"
	"uniform float2 mul_val;\n"
	"uniform float2 add_val;\n"
	"sampler_state def_sampler { Filter = Linear; AddressU = Clamp; AddressV = Clamp; };\n"
	"struct VertInOut { float4 pos : POSITION; float2 uv : TEXCOORD0; };\n"
	"VertInOut VSDefault(VertInOut vert_in) { VertInOut vert_out; vert_out.pos = mul(float4(vert_in.pos.xyz, 1.0), ViewProj); vert_out.uv = vert_in.uv; return vert_out; }\n"
	"float4 PSDefault(VertInOut vert_in) : TARGET { float2 uv = vert_in.uv * mul_val + add_val; return image.Sample(def_sampler, uv); }\n"
	"technique Draw { pass { vertex_shader = VSDefault(vert_in); pixel_shader = PSDefault(vert_in); } }\n";

struct gray_frame {
	uint8_t *data;
	int width;  // detection buffer width
	int height; // detection buffer height
	int stride;
	int src_w; // original source frame width
	int src_h; // original source frame height
	uint64_t frame_seq;
};

struct trackerzoomer_filter {
	obs_source_t *context;
	obs_weak_source_t *parent_weak;

	// AprilTag settings
	bool enable_tracking;

	int tag_id_a;
	int tag_id_b;
	float padding;
	float min_decision_margin; // visual confidence threshold
	int max_hamming;

	int detect_width;

	// Apriltag detector tuning
	float quad_sigma;
	bool refine_edges;
	float decode_sharpening;

	// Detection pacing
	float detect_fps;

	float meas_alpha; // 0..1 measurement smoothing for ROI

	// (release) manual/ROI debug transforms removed

	// auto ROI from detection
	bool auto_roi_valid;
	float auto_roi_cx;
	float auto_roi_cy;
	float auto_roi_w;
	float auto_roi_h;

	// current desired transform (computed from ROI)
	float target_pos_x;
	float target_pos_y;
	float target_scale;

	// smoothed transform (actually applied)
	float cur_pos_x;
	float cur_pos_y;
	float cur_scale;

	// smoothed ROI (measurement smoothing, python parity)
	bool smooth_roi_valid;
	float smooth_roi_cx;
	float smooth_roi_cy;
	float smooth_roi_w;
	float smooth_roi_h;

	// close-tags => force wide/full-frame

	bool _transform_dirty;
	uint64_t _last_apply_ns;
	float _last_applied_pos_x;
	float _last_applied_pos_y;
	float _last_applied_scale;

	// video frame counting
	uint64_t _video_frame_seq;
	int _last_frame_w;
	int _last_frame_h;

	// analysis frame grab pacing
	uint64_t _last_grab_ns;

	// detector thread + shared frame buffer
	pthread_t worker_thread;
	bool worker_running;
#if defined(_WIN32)
	HANDLE worker_event;
#else
	os_event_t *worker_event;
#endif
	pthread_mutex_t frame_mutex;
	pthread_mutex_t td_mutex;
	struct gray_frame pending;
	struct gray_frame work;

	// UI/apply thread safety for transforms
	pthread_mutex_t xform_mutex;
	uint64_t suspend_ui_until_ns;
	float render_mul_x;
	float render_mul_y;
	float render_add_x;
	float render_add_y;

	// (release) debug overlay fields removed

	apriltag_detector_t *td;
	apriltag_family_t *tf;
};

static const char *trackerzoomer_filter_get_name(void *unused)
{
	UNUSED_PARAMETER(unused);
	return "TrackerZoom Filter";
}

static void trackerzoomer_filter_defaults(obs_data_t *settings)
{
	obs_data_set_default_bool(settings, "enable_tracking", false);

	obs_data_set_default_int(settings, "tag_id_a", 0);
	obs_data_set_default_int(settings, "tag_id_b", 1);
	obs_data_set_default_double(settings, "padding", 0.0);
	obs_data_set_default_double(settings, "min_decision_margin", 20.0);
	obs_data_set_default_int(settings, "max_hamming", 0);

	// Detection (external downscale)
	obs_data_set_default_int(settings, "detect_width", 960);
	obs_data_set_default_double(settings, "detect_fps", 30.0);

	// AprilTag detector tuning (minimal)
	obs_data_set_default_double(settings, "quad_sigma", 0.0);
	obs_data_set_default_bool(settings, "refine_edges", true);
	obs_data_set_default_double(settings, "decode_sharpening", 0.25);

	// ROI smoothing
	obs_data_set_default_double(settings, "meas_alpha", 0.2);
}

static void free_gray_frame(struct gray_frame *g)
{
	if (!g)
		return;
	if (g->data)
		bfree(g->data);
	g->data = NULL;
	g->width = 0;
	g->height = 0;
	g->stride = 0;
	g->src_w = 0;
	g->src_h = 0;
	g->frame_seq = 0;
}

static void ensure_gray_frame(struct gray_frame *g, int w, int h)
{
	if (!g)
		return;
	if (w <= 0 || h <= 0)
		return;
	const int stride = w;
	const size_t need = (size_t)stride * (size_t)h;
	if (g->data && g->width == w && g->height == h && g->stride == stride)
		return;
	free_gray_frame(g);
	g->data = bzalloc(need);
	g->width = w;
	g->height = h;
	g->stride = stride;
	g->src_w = 0;
	g->src_h = 0;
}

static inline float clampf(float v, float lo, float hi)
{
	if (v < lo)
		return lo;
	if (v > hi)
		return hi;
	return v;
}

static void draw_solid_rect(gs_effect_t *solid, float x, float y, float w, float h)
{
	if (!solid)
		return;
	if (w <= 0.5f || h <= 0.5f)
		return;

	gs_matrix_push();
	gs_matrix_identity();
	gs_matrix_translate3f(x, y, 0.0f);

	// NOTE: with SOLID effect, texture is ignored; we just draw a colored quad.
	while (gs_effect_loop(solid, "Solid")) {
		gs_draw_sprite(NULL, 0, (uint32_t)lrintf(w), (uint32_t)lrintf(h));
	}
	gs_matrix_pop();
}

static void downscale_luma_nearest(uint8_t *dst, int dst_w, int dst_h, int dst_stride, const uint8_t *src, int src_w,
				   int src_h, int src_stride)
{
	if (!dst || !src || dst_w <= 0 || dst_h <= 0 || src_w <= 0 || src_h <= 0)
		return;

	for (int y = 0; y < dst_h; y++) {
		const int sy = (int)((int64_t)y * src_h / dst_h);
		const uint8_t *src_row = src + sy * src_stride;
		uint8_t *dst_row = dst + y * dst_stride;
		for (int x = 0; x < dst_w; x++) {
			const int sx = (int)((int64_t)x * src_w / dst_w);
			dst_row[x] = src_row[sx];
		}
	}
}

static inline uint8_t rgb_to_luma_u8(int r, int g, int b)
{
	int y8 = (r * 77 + g * 150 + b * 29) >> 8;
	if (y8 < 0)
		y8 = 0;
	if (y8 > 255)
		y8 = 255;
	return (uint8_t)y8;
}

static void bgra_to_gray_downscale(uint8_t *dst, int dst_w, int dst_h, int dst_stride, const uint8_t *src, int src_w,
				   int src_h, int src_stride)
{
	if (!dst || !src)
		return;
	for (int y = 0; y < dst_h; y++) {
		const int sy = (int)((int64_t)y * src_h / dst_h);
		const uint8_t *src_row = src + sy * src_stride;
		uint8_t *dst_row = dst + y * dst_stride;
		for (int x = 0; x < dst_w; x++) {
			const int sx = (int)((int64_t)x * src_w / dst_w);
			const uint8_t *p = src_row + (sx * 4);
			// BGRA
			const int b = p[0];
			const int g = p[1];
			const int r = p[2];
			dst_row[x] = rgb_to_luma_u8(r, g, b);
		}
	}
}

static void rgba_to_gray_downscale(uint8_t *dst, int dst_w, int dst_h, int dst_stride, const uint8_t *src, int src_w,
				   int src_h, int src_stride)
{
	if (!dst || !src)
		return;
	for (int y = 0; y < dst_h; y++) {
		const int sy = (int)((int64_t)y * src_h / dst_h);
		const uint8_t *src_row = src + sy * src_stride;
		uint8_t *dst_row = dst + y * dst_stride;
		for (int x = 0; x < dst_w; x++) {
			const int sx = (int)((int64_t)x * src_w / dst_w);
			const uint8_t *p = src_row + (sx * 4);
			// RGBA
			const int r = p[0];
			const int g = p[1];
			const int b = p[2];
			dst_row[x] = rgb_to_luma_u8(r, g, b);
		}
	}
}

static void bgrx_to_gray_downscale(uint8_t *dst, int dst_w, int dst_h, int dst_stride, const uint8_t *src, int src_w,
				   int src_h, int src_stride)
{
	if (!dst || !src)
		return;
	for (int y = 0; y < dst_h; y++) {
		const int sy = (int)((int64_t)y * src_h / dst_h);
		const uint8_t *src_row = src + sy * src_stride;
		uint8_t *dst_row = dst + y * dst_stride;
		for (int x = 0; x < dst_w; x++) {
			const int sx = (int)((int64_t)x * src_w / dst_w);
			const uint8_t *p = src_row + (sx * 4);
			// BGRX
			const int b = p[0];
			const int g = p[1];
			const int r = p[2];
			dst_row[x] = rgb_to_luma_u8(r, g, b);
		}
	}
}

static void uyvy_to_gray_downscale(uint8_t *dst, int dst_w, int dst_h, int dst_stride, const uint8_t *src, int src_w,
				   int src_h, int src_stride)
{
	// UYVY: U0 Y0 V0 Y1 (2 pixels per 4 bytes)
	if (!dst || !src)
		return;
	for (int y = 0; y < dst_h; y++) {
		const int sy = (int)((int64_t)y * src_h / dst_h);
		const uint8_t *row = src + sy * src_stride;
		uint8_t *out = dst + y * dst_stride;
		for (int x = 0; x < dst_w; x++) {
			const int sx = (int)((int64_t)x * src_w / dst_w);
			const int pair = sx >> 1;
			const int off = pair * 4;
			out[x] = row[off + ((sx & 1) ? 3 : 1)];
		}
	}
}

static void yuy2_to_gray_downscale(uint8_t *dst, int dst_w, int dst_h, int dst_stride, const uint8_t *src, int src_w,
				   int src_h, int src_stride)
{
	// YUY2: Y0 U0 Y1 V0
	if (!dst || !src)
		return;
	for (int y = 0; y < dst_h; y++) {
		const int sy = (int)((int64_t)y * src_h / dst_h);
		const uint8_t *row = src + sy * src_stride;
		uint8_t *out = dst + y * dst_stride;
		for (int x = 0; x < dst_w; x++) {
			const int sx = (int)((int64_t)x * src_w / dst_w);
			const int pair = sx >> 1;
			const int off = pair * 4;
			out[x] = row[off + ((sx & 1) ? 2 : 0)];
		}
	}
}

static void yvyu_to_gray_downscale(uint8_t *dst, int dst_w, int dst_h, int dst_stride, const uint8_t *src, int src_w,
				   int src_h, int src_stride)
{
	// YVYU: Y0 V0 Y1 U0
	if (!dst || !src)
		return;
	for (int y = 0; y < dst_h; y++) {
		const int sy = (int)((int64_t)y * src_h / dst_h);
		const uint8_t *row = src + sy * src_stride;
		uint8_t *out = dst + y * dst_stride;
		for (int x = 0; x < dst_w; x++) {
			const int sx = (int)((int64_t)x * src_w / dst_w);
			const int pair = sx >> 1;
			const int off = pair * 4;
			out[x] = row[off + ((sx & 1) ? 2 : 0)];
		}
	}
}

static void compute_roi_from_detection(const apriltag_detection_t *d, float *out_minx, float *out_miny, float *out_maxx,
				       float *out_maxy)
{
	float minx = d->p[0][0];
	float maxx = d->p[0][0];
	float miny = d->p[0][1];
	float maxy = d->p[0][1];
	for (int i = 1; i < 4; i++) {
		const float x = d->p[i][0];
		const float y = d->p[i][1];
		if (x < minx)
			minx = x;
		if (x > maxx)
			maxx = x;
		if (y < miny)
			miny = y;
		if (y > maxy)
			maxy = y;
	}
	*out_minx = minx;
	*out_miny = miny;
	*out_maxx = maxx;
	*out_maxy = maxy;
}

// Python parity: pick the corner of a tag that points towards the other tag.
static void inner_corner_towards(const apriltag_detection_t *d, float other_cx, float other_cy, float *out_x,
				 float *out_y)
{
	const float cx = d->c[0];
	const float cy = d->c[1];
	float dx = other_cx - cx;
	float dy = other_cy - cy;
	const float norm = sqrtf(dx * dx + dy * dy);
	if (norm <= 1e-9f) {
		*out_x = cx;
		*out_y = cy;
		return;
	}
	dx /= norm;
	dy /= norm;

	int best_i = 0;
	float best_score = -1e30f;
	for (int i = 0; i < 4; i++) {
		const float ox = d->p[i][0] - cx;
		const float oy = d->p[i][1] - cy;
		const float score = ox * dx + oy * dy;
		if (score > best_score) {
			best_score = score;
			best_i = i;
		}
	}
	*out_x = d->p[best_i][0];
	*out_y = d->p[best_i][1];
}

static void update_auto_roi(struct trackerzoomer_filter *f, float minx, float miny, float maxx, float maxy, int det_w,
			    int det_h, int src_w, int src_h, bool inset_padding)
{
	// Python parity (TrackerZoomOBS):
	// - Convert detection bbox into source space
	// - Aspect-fit ROI to OBS canvas aspect ratio
	// - Apply padding as an INSET (never expand beyond fit box)
	// - Clamp ROI center so ROI stays fully inside frame (prevents black)
	// - Clamp ROI size to [min_frac, max_frac] of frame
	if (det_w <= 0 || det_h <= 0 || src_w <= 0 || src_h <= 0) {
		f->auto_roi_valid = false;
		return;
	}

	// Convert detection-space bbox → source-space bbox
	const float sx_det = (float)src_w / (float)det_w;
	const float sy_det = (float)src_h / (float)det_h;
	minx *= sx_det;
	maxx *= sx_det;
	miny *= sy_det;
	maxy *= sy_det;

	// Basic bbox stats in SOURCE space
	float cx = (minx + maxx) * 0.5f;
	float cy = (miny + maxy) * 0.5f;
	float w_box = (maxx - minx);
	float h_box = (maxy - miny);
	if (w_box < 2.0f)
		w_box = 2.0f;
	if (h_box < 2.0f)
		h_box = 2.0f;

	// OBS canvas aspect ratio (fallback to source aspect if unavailable)
	float target_aspect = (float)src_w / (float)src_h;
	struct obs_video_info ovi;
	if (obs_get_video_info(&ovi) && ovi.base_width > 0 && ovi.base_height > 0) {
		target_aspect = (float)ovi.base_width / (float)ovi.base_height;
	}
	if (target_aspect <= 0.0f)
		target_aspect = 16.0f / 9.0f;

	// Aspect-fit (python's _aspect_fit)
	float w_fit = w_box;
	float h_fit = h_box;
	const float box_aspect = w_box / h_box;
	if (box_aspect > target_aspect) {
		// too wide: expand height
		h_fit = w_box / target_aspect;
		w_fit = w_box;
	} else {
		// too tall: expand width
		w_fit = h_box * target_aspect;
		h_fit = h_box;
	}

	// Padding: python insets ROI by padding_px (clamped)
	float padding_px = inset_padding ? f->padding : 0.0f;
	if (padding_px < 0.0f)
		padding_px = 0.0f;
	const float max_padding = (w_fit > 2.0f) ? ((w_fit - 2.0f) * 0.5f) : 0.0f;
	if (padding_px > max_padding)
		padding_px = max_padding;

	float width = w_fit - 2.0f * padding_px;
	float height = width / target_aspect;
	if (width < 2.0f)
		width = 2.0f;
	if (height < 2.0f)
		height = 2.0f;

	// Clamp ROI size to [min_frac, max_frac] of source frame (python defaults)
	const float min_frac = 0.05f;
	const float max_frac = 0.95f;
	const float min_w = (float)src_w * min_frac;
	const float min_h = (float)src_h * min_frac;
	const float max_w = (float)src_w * max_frac;
	const float max_h = (float)src_h * max_frac;
	width = clampf(width, min_w, max_w);
	height = clampf(height, min_h, max_h);

	// Clamp ROI center so ROI stays fully inside frame (python's _clamp_roi_xyxy)
	const float half_w = width * 0.5f;
	const float half_h = height * 0.5f;
	cx = clampf(cx, half_w, (float)src_w - half_w);
	cy = clampf(cy, half_h, (float)src_h - half_h);

	int x1 = (int)lrintf(cx - half_w);
	int y1 = (int)lrintf(cy - half_h);
	int x2 = (int)lrintf(cx + half_w);
	int y2 = (int)lrintf(cy + half_h);
	if (x1 < 0)
		x1 = 0;
	if (y1 < 0)
		y1 = 0;
	if (x2 > src_w)
		x2 = src_w;
	if (y2 > src_h)
		y2 = src_h;

	const float out_w = (float)(x2 - x1);
	const float out_h = (float)(y2 - y1);
	if (out_w < 1.0f || out_h < 1.0f) {
		f->auto_roi_valid = false;
		return;
	}

	f->auto_roi_valid = true;
	f->auto_roi_cx = ((float)x1 + (float)x2) * 0.5f;
	f->auto_roi_cy = ((float)y1 + (float)y2) * 0.5f;
	f->auto_roi_w = out_w;
	f->auto_roi_h = out_h;
	f->_transform_dirty = true;
}

static void *worker_main(void *param)
{
	struct trackerzoomer_filter *f = param;
#if !defined(_WIN32)
	os_set_thread_name("trackerzoomer-apriltag");
#endif

	while (f->worker_running) {
#if defined(_WIN32)
		WaitForSingleObject(f->worker_event, INFINITE);
#else
		os_event_wait(f->worker_event);
#endif
		if (!f->worker_running)
			break;

		// grab latest pending frame
		pthread_mutex_lock(&f->frame_mutex);
		if (f->pending.data && f->pending.frame_seq != f->work.frame_seq) {
			ensure_gray_frame(&f->work, f->pending.width, f->pending.height);
			memcpy(f->work.data, f->pending.data, (size_t)f->pending.stride * (size_t)f->pending.height);
			f->work.frame_seq = f->pending.frame_seq;
			f->work.src_w = f->pending.src_w;
			f->work.src_h = f->pending.src_h;
		}
		pthread_mutex_unlock(&f->frame_mutex);

		if (!f->enable_tracking)
			continue;
		if (!f->work.data || f->work.width <= 0 || f->work.height <= 0)
			continue;

		image_u8_t im = {
			.width = f->work.width,
			.height = f->work.height,
			.stride = f->work.stride,
			.buf = f->work.data,
		};

		pthread_mutex_lock(&f->td_mutex);
		zarray_t *detections = apriltag_detector_detect(f->td, &im);
		pthread_mutex_unlock(&f->td_mutex);
		UNUSED_PARAMETER(zarray_size(detections));
		const apriltag_detection_t *da = NULL;
		const apriltag_detection_t *db = NULL;
		for (int i = 0; i < zarray_size(detections); i++) {
			apriltag_detection_t *d = NULL;
			zarray_get(detections, i, &d);
			if (!d)
				continue;

			// Confidence gating: lower threshold => more detections (more false positives);
			// higher threshold => fewer false positives (more missed detections).
			if (d->decision_margin < f->min_decision_margin)
				continue;

			if (d->hamming > f->max_hamming)
				continue;

			if (d->id == f->tag_id_a)
				da = d;
			else if (d->id == f->tag_id_b)
				db = d;
		}

		// Tracking policy:
		// - 2 tags: update ROI (normal behavior)
		// - 1 tag: assume the other is temporarily obscured -> HOLD last transform (no ROI update)
		// - 0 tags: show full-frame (disable ROI)
		if (da && db) {
			// Python parity: ROI is based on the INNER corners facing each other,
			// so the markers themselves get cropped out of shot.
			float ax, ay, bx, by;
			inner_corner_towards(da, db->c[0], db->c[1], &ax, &ay);
			inner_corner_towards(db, da->c[0], da->c[1], &bx, &by);

			float minx = (ax < bx) ? ax : bx;
			float miny = (ay < by) ? ay : by;
			float maxx = (ax > bx) ? ax : bx;
			float maxy = (ay > by) ? ay : by;

			// Compute average tag edge length (px) as a scale reference.
			const float a_e0 = hypotf(da->p[1][0] - da->p[0][0], da->p[1][1] - da->p[0][1]);
			const float a_e1 = hypotf(da->p[2][0] - da->p[1][0], da->p[2][1] - da->p[1][1]);
			const float a_e2 = hypotf(da->p[3][0] - da->p[2][0], da->p[3][1] - da->p[2][1]);
			const float a_e3 = hypotf(da->p[0][0] - da->p[3][0], da->p[0][1] - da->p[3][1]);
			const float b_e0 = hypotf(db->p[1][0] - db->p[0][0], db->p[1][1] - db->p[0][1]);
			const float b_e1 = hypotf(db->p[2][0] - db->p[1][0], db->p[2][1] - db->p[1][1]);
			const float b_e2 = hypotf(db->p[3][0] - db->p[2][0], db->p[3][1] - db->p[2][1]);
			const float b_e3 = hypotf(db->p[0][0] - db->p[3][0], db->p[0][1] - db->p[3][1]);

			const float a_avg = (a_e0 + a_e1 + a_e2 + a_e3) * 0.25f;
			const float b_avg = (b_e0 + b_e1 + b_e2 + b_e3) * 0.25f;
			float avg_edge = (a_avg + b_avg) * 0.5f;
			if (!isfinite(avg_edge) || avg_edge <= 0.0f)
				avg_edge = 0.0f;

			// New feature: if tags are very close, switch to full-frame ("zoom off").
			// Threshold: one tag width (we approximate with avg edge length).
			float min_corner_dist = 1e30f;
			for (int i = 0; i < 4; i++) {
				for (int j = 0; j < 4; j++) {
					const float dx = da->p[i][0] - db->p[j][0];
					const float dy = da->p[i][1] - db->p[j][1];
					const float d = sqrtf(dx * dx + dy * dy);
					if (d < min_corner_dist)
						min_corner_dist = d;
				}
			}
			// Close-tags => wide/full-frame (zoom off) gesture.
			// Threshold: one tag width (we approximate with avg edge length).
			const bool too_close = (avg_edge > 0.0f) && isfinite(min_corner_dist) &&
					       (min_corner_dist < avg_edge);

			// Auto-inset to remove the white border around printed tags.
			// Spec: avg edge length (px) / 6, plus user padding.
			const float auto_inset = avg_edge / 6.0f;

			// Temporarily add auto_inset to f->padding for this ROI update.
			const float user_pad = f->padding;
			f->padding = user_pad + auto_inset;

			pthread_mutex_lock(&f->frame_mutex);
			if (too_close) {
				// Force full-frame by invalidating ROI.
				f->auto_roi_valid = false;
				f->_transform_dirty = true;
			} else {
				update_auto_roi(f, minx, miny, maxx, maxy, f->work.width, f->work.height, f->work.src_w,
						f->work.src_h, true);
			}
			pthread_mutex_unlock(&f->frame_mutex);

			// Restore user padding.
			f->padding = user_pad;
		} else if (!da && !db) {
			pthread_mutex_lock(&f->frame_mutex);
			f->auto_roi_valid = false;
			f->_transform_dirty = true;
			pthread_mutex_unlock(&f->frame_mutex);
		}
		// else: exactly one tag -> do nothing (hold last ROI/transform)

		apriltag_detections_destroy(detections);
	}

	return NULL;
}

static void *trackerzoomer_filter_create(obs_data_t *settings, obs_source_t *source)
{
	struct trackerzoomer_filter *f = bzalloc(sizeof(*f));
	f->context = source;
	f->_last_applied_pos_x = 0.0f;
	f->_last_applied_pos_y = 0.0f;
	f->_last_applied_scale = 1.0f;
	f->target_pos_x = 0.0f;
	f->target_pos_y = 0.0f;
	f->target_scale = 1.0f;
	f->cur_pos_x = 0.0f;
	f->cur_pos_y = 0.0f;
	f->cur_scale = 1.0f;

	f->smooth_roi_valid = false;
	f->smooth_roi_cx = 0.0f;
	f->smooth_roi_cy = 0.0f;
	f->smooth_roi_w = 0.0f;
	f->smooth_roi_h = 0.0f;
	// (removed) close_tags_wide
	pthread_mutex_init_value(&f->frame_mutex);
	pthread_mutex_init_value(&f->td_mutex);
	pthread_mutex_init_value(&f->xform_mutex);
	f->suspend_ui_until_ns = 0;
	f->render_mul_x = 1.0f;
	f->render_mul_y = 1.0f;
	f->render_add_x = 0.0f;
	f->render_add_y = 0.0f;
	// (release) debug overlays removed

	// detector setup (tag16h5)
	f->tf = tag16h5_create();
	f->td = apriltag_detector_create();
	apriltag_detector_add_family_bits(f->td, f->tf, 1);
	f->td->nthreads = 2;
	// Defaults; actual values are set in update() from UI.
	f->td->quad_decimate = 1.0f;
	f->td->quad_sigma = 0.0f;
	f->td->refine_edges = 1;
	f->td->decode_sharpening = 0.25f;
	f->td->qtp.deglitch = 0;

	// worker thread
#if defined(_WIN32)
	f->worker_event = CreateEventW(NULL, FALSE, FALSE, NULL); // auto-reset
#else
	os_event_init(&f->worker_event, OS_EVENT_TYPE_AUTO);
#endif
	f->worker_running = true;
	pthread_create(&f->worker_thread, NULL, worker_main, f);

	obs_source_update(source, settings);
	return f;
}

static void trackerzoomer_filter_destroy(void *data)
{
	struct trackerzoomer_filter *f = data;
	if (!f)
		return;

	// stop worker
	f->worker_running = false;
#if defined(_WIN32)
	if (f->worker_event)
		SetEvent(f->worker_event);
#else
	if (f->worker_event)
		os_event_signal(f->worker_event);
#endif
	pthread_join(f->worker_thread, NULL);
#if defined(_WIN32)
	if (f->worker_event)
		CloseHandle(f->worker_event);
#else
	if (f->worker_event)
		os_event_destroy(f->worker_event);
#endif

	pthread_mutex_lock(&f->frame_mutex);
	free_gray_frame(&f->pending);
	free_gray_frame(&f->work);
	pthread_mutex_unlock(&f->frame_mutex);

	if (f->td)
		apriltag_detector_destroy(f->td);
	if (f->tf)
		tag16h5_destroy(f->tf);

	if (f->parent_weak)
		obs_weak_source_release(f->parent_weak);

	pthread_mutex_destroy(&f->frame_mutex);
	pthread_mutex_destroy(&f->td_mutex);
	pthread_mutex_destroy(&f->xform_mutex);

	bfree(f);
}

static obs_properties_t *trackerzoomer_filter_properties(void *data)
{
	UNUSED_PARAMETER(data);
	obs_properties_t *props = obs_properties_create();

	obs_properties_add_bool(props, "enable_tracking", "Enable AprilTag tracking");

	obs_properties_add_int(props, "tag_id_a", "Tag ID A", 0, 1, 1);
	obs_properties_add_int(props, "tag_id_b", "Tag ID B", 0, 1, 1);
	obs_properties_add_float(props, "padding", "Padding (px)", 0.0, 500.0, 1.0);
	obs_properties_add_float_slider(props, "min_decision_margin", "Min decision margin", 0.0, 100.0, 1.0);
	obs_properties_add_int_slider(props, "max_hamming", "Max hamming", 0, 2, 1);

	obs_properties_add_int_slider(props, "detect_width", "Detection width (px)", 160, 3840, 16);
	obs_properties_add_float_slider(props, "detect_fps", "Detection FPS", 1.0, 60.0, 1.0);

	obs_properties_add_float_slider(props, "quad_sigma", "quad_sigma", 0.0, 2.0, 0.05);
	obs_properties_add_bool(props, "refine_edges", "refine_edges");
	obs_properties_add_float_slider(props, "decode_sharpening", "decode_sharpening", 0.0, 1.0, 0.05);

	obs_properties_add_float_slider(props, "meas_alpha", "ROI smoothing (alpha)", 0.0, 1.0, 0.01);

	return props;
}

static void trackerzoomer_filter_update(void *data, obs_data_t *settings)
{
	struct trackerzoomer_filter *f = data;
	if (!f)
		return;

	const bool prev_enable = f->enable_tracking;
	f->enable_tracking = obs_data_get_bool(settings, "enable_tracking");

	// Avoid deadlocking with OBS settings save: pause UI transform tasks briefly on any update.
	// In particular, enable/disable tracking triggers an immediate save.
	const uint64_t now_ns = os_gettime_ns();
	f->suspend_ui_until_ns = now_ns + (prev_enable != f->enable_tracking ? 1000000000ULL : 200000000ULL);

	f->tag_id_a = (int)obs_data_get_int(settings, "tag_id_a");
	f->tag_id_b = (int)obs_data_get_int(settings, "tag_id_b");

	f->padding = (float)obs_data_get_double(settings, "padding");
	if (f->padding < 0.0f)
		f->padding = 0.0f;

	f->min_decision_margin = (float)obs_data_get_double(settings, "min_decision_margin");
	if (f->min_decision_margin < 0.0f)
		f->min_decision_margin = 0.0f;

	f->max_hamming = (int)obs_data_get_int(settings, "max_hamming");
	if (f->max_hamming < 0)
		f->max_hamming = 0;
	if (f->max_hamming > 2)
		f->max_hamming = 2;

	f->detect_width = (int)obs_data_get_int(settings, "detect_width");
	if (f->detect_width < 160)
		f->detect_width = 160;
	if (f->detect_width > 3840)
		f->detect_width = 3840;

	f->detect_fps = (float)obs_data_get_double(settings, "detect_fps");
	if (f->detect_fps < 1.0f)
		f->detect_fps = 1.0f;

	f->quad_sigma = (float)obs_data_get_double(settings, "quad_sigma");
	f->refine_edges = obs_data_get_bool(settings, "refine_edges");
	f->decode_sharpening = (float)obs_data_get_double(settings, "decode_sharpening");

	f->meas_alpha = (float)obs_data_get_double(settings, "meas_alpha");
	if (f->meas_alpha < 0.0f)
		f->meas_alpha = 0.0f;
	if (f->meas_alpha > 1.0f)
		f->meas_alpha = 1.0f;

	// Apply detector params (worker thread also uses td).
	pthread_mutex_lock(&f->td_mutex);
	if (f->td) {
		f->td->quad_decimate = 1.0f; // we external-downscale via detect_width
		f->td->quad_sigma = clampf(f->quad_sigma, 0.0f, 2.0f);
		f->td->refine_edges = f->refine_edges ? 1 : 0;
		f->td->decode_sharpening = clampf(f->decode_sharpening, 0.0f, 1.0f);
	}
	pthread_mutex_unlock(&f->td_mutex);
}

static void feed_pending_from_frame(struct trackerzoomer_filter *f, struct obs_source_frame *frame)
{
	if (!f || !frame)
		return;

	// Track basic frame characteristics for logging.
	pthread_mutex_lock(&f->frame_mutex);
	// (release) frame format diagnostics removed
	f->_last_frame_w = (int)frame->width;
	f->_last_frame_h = (int)frame->height;
	pthread_mutex_unlock(&f->frame_mutex);

	const int src_w = (int)frame->width;
	const int src_h = (int)frame->height;
	if (src_w <= 0 || src_h <= 0)
		return;

	int dst_w = f->detect_width;
	if (dst_w < 1)
		dst_w = 1;
	if (src_w < dst_w)
		dst_w = src_w;
	int dst_h = (int)((int64_t)dst_w * src_h / src_w);
	if (dst_h < 1)
		dst_h = 1;

	pthread_mutex_lock(&f->frame_mutex);
	ensure_gray_frame(&f->pending, dst_w, dst_h);
	f->pending.src_w = src_w;
	f->pending.src_h = src_h;

	if (frame->format == VIDEO_FORMAT_I420 || frame->format == VIDEO_FORMAT_NV12 ||
	    frame->format == VIDEO_FORMAT_Y800 || frame->format == VIDEO_FORMAT_I444 ||
	    frame->format == VIDEO_FORMAT_I422) {
		// For planar formats (and grayscale), the first plane is luma or intensity.
		const uint8_t *yplane = frame->data[0];
		const int ystride = (int)frame->linesize[0];
		downscale_luma_nearest(f->pending.data, dst_w, dst_h, f->pending.stride, yplane, src_w, src_h, ystride);
		f->pending.frame_seq = f->_video_frame_seq;
#if defined(_WIN32)
		SetEvent(f->worker_event);
#else
		os_event_signal(f->worker_event);
#endif
	} else if (frame->format == VIDEO_FORMAT_UYVY) {
		const uint8_t *src = frame->data[0];
		const int stride = (int)frame->linesize[0];
		uyvy_to_gray_downscale(f->pending.data, dst_w, dst_h, f->pending.stride, src, src_w, src_h, stride);
		f->pending.frame_seq = f->_video_frame_seq;
#if defined(_WIN32)
		SetEvent(f->worker_event);
#else
		os_event_signal(f->worker_event);
#endif
	} else if (frame->format == VIDEO_FORMAT_YUY2) {
		const uint8_t *src = frame->data[0];
		const int stride = (int)frame->linesize[0];
		yuy2_to_gray_downscale(f->pending.data, dst_w, dst_h, f->pending.stride, src, src_w, src_h, stride);
		f->pending.frame_seq = f->_video_frame_seq;
#if defined(_WIN32)
		SetEvent(f->worker_event);
#else
		os_event_signal(f->worker_event);
#endif
	} else if (frame->format == VIDEO_FORMAT_YVYU) {
		const uint8_t *src = frame->data[0];
		const int stride = (int)frame->linesize[0];
		yvyu_to_gray_downscale(f->pending.data, dst_w, dst_h, f->pending.stride, src, src_w, src_h, stride);
		f->pending.frame_seq = f->_video_frame_seq;
#if defined(_WIN32)
		SetEvent(f->worker_event);
#else
		os_event_signal(f->worker_event);
#endif
	} else if (frame->format == VIDEO_FORMAT_BGRA) {
		const uint8_t *src = frame->data[0];
		const int stride = (int)frame->linesize[0];
		bgra_to_gray_downscale(f->pending.data, dst_w, dst_h, f->pending.stride, src, src_w, src_h, stride);
		f->pending.frame_seq = f->_video_frame_seq;
#if defined(_WIN32)
		SetEvent(f->worker_event);
#else
		os_event_signal(f->worker_event);
#endif
	} else if (frame->format == VIDEO_FORMAT_RGBA) {
		const uint8_t *src = frame->data[0];
		const int stride = (int)frame->linesize[0];
		rgba_to_gray_downscale(f->pending.data, dst_w, dst_h, f->pending.stride, src, src_w, src_h, stride);
		f->pending.frame_seq = f->_video_frame_seq;
#if defined(_WIN32)
		SetEvent(f->worker_event);
#else
		os_event_signal(f->worker_event);
#endif
	} else if (frame->format == VIDEO_FORMAT_BGRX) {
		const uint8_t *src = frame->data[0];
		const int stride = (int)frame->linesize[0];
		bgrx_to_gray_downscale(f->pending.data, dst_w, dst_h, f->pending.stride, src, src_w, src_h, stride);
		f->pending.frame_seq = f->_video_frame_seq;
#if defined(_WIN32)
		SetEvent(f->worker_event);
#else
		os_event_signal(f->worker_event);
#endif
	} else {
		// Unsupported format for now
	}

	pthread_mutex_unlock(&f->frame_mutex);
}

static void trackerzoomer_filter_tick(void *data, float seconds)
{
	UNUSED_PARAMETER(seconds);
	struct trackerzoomer_filter *f = data;
	if (!f)
		return;

	// When tracking is off, render full frame.
	if (!f->enable_tracking) {
		pthread_mutex_lock(&f->xform_mutex);
		f->render_mul_x = 1.0f;
		f->render_mul_y = 1.0f;
		f->render_add_x = 0.0f;
		f->render_add_y = 0.0f;
		pthread_mutex_unlock(&f->xform_mutex);
		return;
	}

	// Determine ROI from worker-produced auto ROI (smoothed).
	bool raw_valid = false;
	float raw_cx = 0.0f, raw_cy = 0.0f, raw_w = 0.0f, raw_h = 0.0f;
	pthread_mutex_lock(&f->frame_mutex);
	raw_valid = f->auto_roi_valid;
	if (raw_valid) {
		raw_cx = f->auto_roi_cx;
		raw_cy = f->auto_roi_cy;
		raw_w = f->auto_roi_w;
		raw_h = f->auto_roi_h;
	}
	pthread_mutex_unlock(&f->frame_mutex);

	float roi_cx = 0.0f, roi_cy = 0.0f, roi_w = 0.0f, roi_h = 0.0f;
	bool use_roi = false;
	if (raw_valid) {
		const float m_alpha = f->meas_alpha;
		if (!f->smooth_roi_valid) {
			f->smooth_roi_valid = true;
			f->smooth_roi_cx = raw_cx;
			f->smooth_roi_cy = raw_cy;
			f->smooth_roi_w = raw_w;
			f->smooth_roi_h = raw_h;
		} else {
			f->smooth_roi_cx += (raw_cx - f->smooth_roi_cx) * m_alpha;
			f->smooth_roi_cy += (raw_cy - f->smooth_roi_cy) * m_alpha;
			f->smooth_roi_w += (raw_w - f->smooth_roi_w) * m_alpha;
			f->smooth_roi_h += (raw_h - f->smooth_roi_h) * m_alpha;
		}
		roi_cx = f->smooth_roi_cx;
		roi_cy = f->smooth_roi_cy;
		roi_w = f->smooth_roi_w;
		roi_h = f->smooth_roi_h;
		use_roi = true;
	} else {
		f->smooth_roi_valid = false;
	}

	// Convert ROI to UV crop parameters.
	float mul_x = 1.0f, mul_y = 1.0f, add_x = 0.0f, add_y = 0.0f;
	if (use_roi && f->_last_frame_w > 0 && f->_last_frame_h > 0) {
		const float fw = (float)f->_last_frame_w;
		const float fh = (float)f->_last_frame_h;
		if (roi_w < 1.0f)
			roi_w = 1.0f;
		if (roi_h < 1.0f)
			roi_h = 1.0f;
		float u0 = (roi_cx - roi_w * 0.5f) / fw;
		float v0 = (roi_cy - roi_h * 0.5f) / fh;
		float u1 = (roi_cx + roi_w * 0.5f) / fw;
		float v1 = (roi_cy + roi_h * 0.5f) / fh;
		// clamp
		u0 = clampf(u0, 0.0f, 1.0f);
		v0 = clampf(v0, 0.0f, 1.0f);
		u1 = clampf(u1, 0.0f, 1.0f);
		v1 = clampf(v1, 0.0f, 1.0f);
		mul_x = (u1 - u0);
		mul_y = (v1 - v0);
		add_x = u0;
		add_y = v0;
		if (mul_x < 1e-4f || mul_y < 1e-4f) {
			mul_x = 1.0f;
			mul_y = 1.0f;
			add_x = 0.0f;
			add_y = 0.0f;
		}
	}

	pthread_mutex_lock(&f->xform_mutex);
	f->render_mul_x = mul_x;
	f->render_mul_y = mul_y;
	f->render_add_x = add_x;
	f->render_add_y = add_y;
	pthread_mutex_unlock(&f->xform_mutex);
}

static struct obs_source_frame *trackerzoomer_filter_video(void *data, struct obs_source_frame *frame)
{
	// IMPORTANT: never mutate or replace the incoming frame; we only *observe*
	// it for analysis and return it untouched.
	struct trackerzoomer_filter *f = data;
	if (!f || !frame)
		return frame;

	// Don't query parent source here; keep this callback as lightweight as possible.

	// Preferred path: get analysis frames from the filter_video callback.
	// This avoids polling the parent source, which can degrade macOS webcam feed
	// quality (stutter/glitches) due to internal locking/conversion.
	if (f->enable_tracking) {
		const uint64_t now = os_gettime_ns();
		const float fps = (f->detect_fps > 0.1f) ? f->detect_fps : 30.0f;
		const uint64_t interval = (uint64_t)(1000000000.0 / (double)fps);

		if (!f->_last_grab_ns || (now - f->_last_grab_ns) >= interval) {
			f->_last_grab_ns = now;
			f->_video_frame_seq++;
			feed_pending_from_frame(f, frame);
		}
	}

	return frame;
}

static void trackerzoomer_filter_video_render(void *data, gs_effect_t *effect)
{
	UNUSED_PARAMETER(effect);
	struct trackerzoomer_filter *f = data;
	if (!f)
		return;

	if (!g_trackerzoomer_crop_effect) {
		g_trackerzoomer_crop_effect = gs_effect_create(k_trackerzoomer_crop_effect_src, NULL, NULL);
		if (g_trackerzoomer_crop_effect) {
			g_param_mul = gs_effect_get_param_by_name(g_trackerzoomer_crop_effect, "mul_val");
			g_param_add = gs_effect_get_param_by_name(g_trackerzoomer_crop_effect, "add_val");
		}
	}

	gs_effect_t *fx = g_trackerzoomer_crop_effect ? g_trackerzoomer_crop_effect
						      : obs_get_base_effect(OBS_EFFECT_DEFAULT);
	if (!fx) {
		obs_source_skip_video_filter(f->context);
		return;
	}

	float mul_x = 1.0f, mul_y = 1.0f, add_x = 0.0f, add_y = 0.0f;
	pthread_mutex_lock(&f->xform_mutex);
	mul_x = f->render_mul_x;
	mul_y = f->render_mul_y;
	add_x = f->render_add_x;
	add_y = f->render_add_y;
	pthread_mutex_unlock(&f->xform_mutex);

	const bool began = obs_source_process_filter_begin(f->context, GS_RGBA, 0);
	if (!began) {
		obs_source_skip_video_filter(f->context);
		return;
	}

	if (fx == g_trackerzoomer_crop_effect && g_param_mul && g_param_add) {
		struct vec2 mul = {mul_x, mul_y};
		struct vec2 add = {add_x, add_y};
		gs_effect_set_vec2(g_param_mul, &mul);
		gs_effect_set_vec2(g_param_add, &add);
	}

	obs_source_process_filter_end(f->context, fx, 0, 0);
}

static struct obs_source_info trackerzoomer_filter_info = {
	.id = "trackerzoomer_filter",
	.type = OBS_SOURCE_TYPE_FILTER,
	// Rendered video filter.
	.output_flags = OBS_SOURCE_VIDEO,
	.get_name = trackerzoomer_filter_get_name,
	.get_defaults = trackerzoomer_filter_defaults,
	.create = trackerzoomer_filter_create,
	.destroy = trackerzoomer_filter_destroy,
	.get_properties = trackerzoomer_filter_properties,
	.update = trackerzoomer_filter_update,
	.video_tick = trackerzoomer_filter_tick,
	.filter_video = trackerzoomer_filter_video,
	.video_render = trackerzoomer_filter_video_render,
};

void register_trackerzoomer_filter(void)
{
	obs_register_source(&trackerzoomer_filter_info);
}
