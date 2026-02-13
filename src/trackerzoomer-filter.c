#include <obs-module.h>
#include <graphics/graphics.h>
#include <util/platform.h>
#include <util/threading.h>

#if defined(HAVE_OBS_FRONTEND_API)
#include <obs/obs-frontend-api.h>
#endif

#include <stdio.h>
#include <math.h>

#include "apriltag.h"
#include "tag16h5.h"

struct gray_frame {
	uint8_t *data;
	int width;   // detection buffer width
	int height;  // detection buffer height
	int stride;
	int src_w;   // original source frame width
	int src_h;   // original source frame height
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

	// Detection image scaling / apriltag internal decimation
	// 0 = external downscale to detect_width; 1 = feed full-res + use td->quad_decimate
	int downscale_mode;
	int detect_width;

	// Apriltag detector tuning
	float quad_decimate;
	float quad_sigma;
	bool refine_edges;
	float decode_sharpening;
	bool deglitch;
	int min_cluster_pixels;
	int max_nmaxima;
	float critical_rad;       // radians
	float max_line_fit_mse;
	int min_white_black_diff;

	// Detection pacing
	float detect_fps;

	// Transform jump guards / smoothing
	bool freeze_transform;
	bool apply_translation;
	bool apply_scale;
	float meas_alpha;            // 0..1 measurement smoothing for ROI
	float ease_tau;              // seconds, transform easing time constant
	bool jump_guard;
	float max_pos_jump_px;       // max change in *target* pos per apply step
	float max_scale_jump;        // max change in *target* scale per apply step

	// canvas clamp (prevents black borders when tracking near frame edges)
	bool clamp_to_canvas;
	float clamp_margin_px;

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
	uint64_t _last_log_ns;
	int _last_frame_format;
	int _last_frame_w;
	int _last_frame_h;
	uint64_t _last_dump_ns;
	int _dump_count;

	// analysis frame grab pacing (tick thread)
	uint64_t _last_grab_ns;

	// render-path forensics
	uint64_t _last_render_log_ns;
	int _last_render_parent_w;
	int _last_render_parent_h;

	// detection stats (worker thread)
	uint64_t _last_detect_ns;
	uint64_t _last_detect_trigger_ns;
	int _last_detect_count;
	bool _last_found_a;
	bool _last_found_b;

	// detector thread + shared frame buffer
	pthread_t worker_thread;
	bool worker_running;
	os_event_t *worker_event;
	pthread_mutex_t frame_mutex;
	pthread_mutex_t td_mutex;
	struct gray_frame pending;
	struct gray_frame work;

	// UI/apply thread safety for transforms
	pthread_mutex_t xform_mutex;
	float apply_pos_x;
	float apply_pos_y;
	float apply_scale_val;

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

	obs_data_set_default_int(settings, "downscale_mode", 0); // 0 external, 1 apriltag quad_decimate
	obs_data_set_default_int(settings, "detect_width", 960);

	// Apriltag tuning defaults (conservative)
	obs_data_set_default_double(settings, "quad_decimate", 1.0);
	obs_data_set_default_double(settings, "quad_sigma", 0.0);
	obs_data_set_default_bool(settings, "refine_edges", true);
	obs_data_set_default_double(settings, "decode_sharpening", 0.25);
	obs_data_set_default_bool(settings, "deglitch", false);
	obs_data_set_default_int(settings, "min_cluster_pixels", 5);
	obs_data_set_default_int(settings, "max_nmaxima", 10);
	obs_data_set_default_double(settings, "critical_rad", 0.0);
	obs_data_set_default_double(settings, "max_line_fit_mse", 10.0);
	obs_data_set_default_int(settings, "min_white_black_diff", 5);

	obs_data_set_default_double(settings, "detect_fps", 30.0);

	// Transform debugging / stabilization
	obs_data_set_default_bool(settings, "freeze_transform", false);
	obs_data_set_default_bool(settings, "apply_translation", true);
	obs_data_set_default_bool(settings, "apply_scale", true);
	obs_data_set_default_double(settings, "meas_alpha", 0.2);
	obs_data_set_default_double(settings, "ease_tau", 0.5);
	obs_data_set_default_bool(settings, "jump_guard", false);
	obs_data_set_default_double(settings, "max_pos_jump_px", 150.0);
	obs_data_set_default_double(settings, "max_scale_jump", 0.5);

	// Debug overlay (ROI preview)
	// (release) debug overlay defaults removed

	// When enabled, clamps the scene-item translation so the scaled source still
	// covers the canvas (no black bars revealed behind it).
	obs_data_set_default_bool(settings, "clamp_to_canvas", true);
	obs_data_set_default_double(settings, "clamp_margin_px", 0.0);

	// (release) manual/ROI debug transform defaults removed
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

static void downscale_luma_nearest(uint8_t *dst, int dst_w, int dst_h, int dst_stride, const uint8_t *src, int src_w, int src_h, int src_stride)
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

static void bgra_to_gray_downscale(uint8_t *dst, int dst_w, int dst_h, int dst_stride, const uint8_t *src, int src_w, int src_h, int src_stride)
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

static void rgba_to_gray_downscale(uint8_t *dst, int dst_w, int dst_h, int dst_stride, const uint8_t *src, int src_w, int src_h, int src_stride)
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

static void bgrx_to_gray_downscale(uint8_t *dst, int dst_w, int dst_h, int dst_stride, const uint8_t *src, int src_w, int src_h, int src_stride)
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

static void dump_pgm_u8(const char *path, const uint8_t *buf, int w, int h, int stride)
{
	if (!path || !buf || w <= 0 || h <= 0 || stride < w)
		return;
	FILE *fp = fopen(path, "wb");
	if (!fp)
		return;
	fprintf(fp, "P5\n%d %d\n255\n", w, h);
	for (int y = 0; y < h; y++)
		fwrite(buf + y * stride, 1, (size_t)w, fp);
	fclose(fp);
}

static void uyvy_to_gray_downscale(uint8_t *dst, int dst_w, int dst_h, int dst_stride, const uint8_t *src, int src_w, int src_h,
				  int src_stride)
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

static void yuy2_to_gray_downscale(uint8_t *dst, int dst_w, int dst_h, int dst_stride, const uint8_t *src, int src_w, int src_h,
				  int src_stride)
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

static void yvyu_to_gray_downscale(uint8_t *dst, int dst_w, int dst_h, int dst_stride, const uint8_t *src, int src_w, int src_h,
				  int src_stride)
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

static void compute_roi_from_detection(const apriltag_detection_t *d, float *out_minx, float *out_miny, float *out_maxx, float *out_maxy)
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
static void inner_corner_towards(const apriltag_detection_t *d, float other_cx, float other_cy, float *out_x, float *out_y)
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

static void update_auto_roi(struct trackerzoomer_filter *f, float minx, float miny, float maxx, float maxy, int det_w, int det_h,
			    int src_w, int src_h, bool inset_padding)
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
	os_set_thread_name("trackerzoomer-apriltag");

	while (f->worker_running) {
		os_event_wait(f->worker_event);
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
		const int det_count = zarray_size(detections);
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
			const bool too_close = (avg_edge > 0.0f) && isfinite(min_corner_dist) && (min_corner_dist < avg_edge);

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
				update_auto_roi(f, minx, miny, maxx, maxy, f->work.width, f->work.height, f->work.src_w, f->work.src_h, true);
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

		pthread_mutex_lock(&f->frame_mutex);
		f->_last_detect_ns = os_gettime_ns();
		f->_last_detect_count = det_count;
		f->_last_found_a = (da != NULL);
		f->_last_found_b = (db != NULL);
		pthread_mutex_unlock(&f->frame_mutex);

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
	f->apply_pos_x = 0.0f;
	f->apply_pos_y = 0.0f;
	f->apply_scale_val = 1.0f;
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
	os_event_init(&f->worker_event, OS_EVENT_TYPE_AUTO);
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
	if (f->worker_event)
		os_event_signal(f->worker_event);
	pthread_join(f->worker_thread, NULL);
	if (f->worker_event)
		os_event_destroy(f->worker_event);

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

	// Detection scaling mode (diagnostic)
	obs_property_t *p_mode = obs_properties_add_list(props, "downscale_mode", "Detection scaling mode",
							OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(p_mode, "External downscale (detect_width)", 0);
	obs_property_list_add_int(p_mode, "Apriltag quad_decimate (feed full-res)", 1);
	obs_properties_add_int_slider(props, "detect_width", "External detect width (px)", 160, 3840, 16);

	// Apriltag detector tuning
	obs_properties_add_float_slider(props, "quad_decimate", "quad_decimate", 1.0, 6.0, 0.1);
	obs_properties_add_float_slider(props, "quad_sigma", "quad_sigma", 0.0, 2.0, 0.05);
	obs_properties_add_bool(props, "refine_edges", "refine_edges");
	obs_properties_add_float_slider(props, "decode_sharpening", "decode_sharpening", 0.0, 1.0, 0.05);
	obs_properties_add_bool(props, "deglitch", "qtp.deglitch");
	obs_properties_add_int_slider(props, "min_cluster_pixels", "qtp.min_cluster_pixels", 0, 1000, 1);
	obs_properties_add_int_slider(props, "max_nmaxima", "qtp.max_nmaxima", 1, 100, 1);
	obs_properties_add_float_slider(props, "critical_rad", "qtp.critical_rad (rad)", 0.0, 3.14159, 0.01);
	obs_properties_add_float_slider(props, "max_line_fit_mse", "qtp.max_line_fit_mse", 0.0, 100.0, 0.5);
	obs_properties_add_int_slider(props, "min_white_black_diff", "qtp.min_white_black_diff", 0, 255, 1);

	obs_properties_add_float_slider(props, "detect_fps", "Detection FPS", 1.0, 60.0, 1.0);

	// (release) analysis frame source fixed to filter_video callback

	// Transform debugging / stabilization
	obs_properties_add_bool(props, "freeze_transform", "Freeze transform (still detect, don't apply)");
	obs_properties_add_bool(props, "apply_translation", "Apply translation");
	obs_properties_add_bool(props, "apply_scale", "Apply scale");
	obs_properties_add_float_slider(props, "meas_alpha", "ROI measurement smoothing (alpha)", 0.0, 1.0, 0.01);
	obs_properties_add_float_slider(props, "ease_tau", "Transform easing tau (s)", 0.01, 2.0, 0.01);
	obs_properties_add_bool(props, "jump_guard", "Jump guard (clamp per-step changes)");
	obs_properties_add_float_slider(props, "max_pos_jump_px", "Max pos jump per step (px)", 0.0, 2000.0, 5.0);
	obs_properties_add_float_slider(props, "max_scale_jump", "Max scale jump per step", 0.0, 5.0, 0.01);

	// Release build: no debug overlays

	obs_properties_add_bool(props, "clamp_to_canvas", "Clamp to canvas (avoid black borders)");
	obs_properties_add_float_slider(props, "clamp_margin_px", "Clamp margin (px)", 0.0, 200.0, 1.0);

	return props;
}

static void trackerzoomer_filter_update(void *data, obs_data_t *settings)
{
	struct trackerzoomer_filter *f = data;
	if (!f)
		return;

	f->enable_tracking = obs_data_get_bool(settings, "enable_tracking");
	f->tag_id_a = (int)obs_data_get_int(settings, "tag_id_a");
	f->tag_id_b = (int)obs_data_get_int(settings, "tag_id_b");
	f->padding = (float)obs_data_get_double(settings, "padding");
	f->min_decision_margin = (float)obs_data_get_double(settings, "min_decision_margin");
	if (f->min_decision_margin < 0.0f)
		f->min_decision_margin = 0.0f;
	f->max_hamming = (int)obs_data_get_int(settings, "max_hamming");
	if (f->max_hamming < 0)
		f->max_hamming = 0;
	if (f->max_hamming > 2)
		f->max_hamming = 2;

	f->downscale_mode = (int)obs_data_get_int(settings, "downscale_mode");
	f->detect_width = (int)obs_data_get_int(settings, "detect_width");
	if (f->detect_width < 160)
		f->detect_width = 160;
	if (f->detect_width > 3840)
		f->detect_width = 3840;

	f->quad_decimate = (float)obs_data_get_double(settings, "quad_decimate");
	f->quad_sigma = (float)obs_data_get_double(settings, "quad_sigma");
	f->refine_edges = obs_data_get_bool(settings, "refine_edges");
	f->decode_sharpening = (float)obs_data_get_double(settings, "decode_sharpening");
	f->deglitch = obs_data_get_bool(settings, "deglitch");
	f->min_cluster_pixels = (int)obs_data_get_int(settings, "min_cluster_pixels");
	f->max_nmaxima = (int)obs_data_get_int(settings, "max_nmaxima");
	f->critical_rad = (float)obs_data_get_double(settings, "critical_rad");
	f->max_line_fit_mse = (float)obs_data_get_double(settings, "max_line_fit_mse");
	f->min_white_black_diff = (int)obs_data_get_int(settings, "min_white_black_diff");

	// (release) detect_every_n_frames removed
	f->detect_fps = (float)obs_data_get_double(settings, "detect_fps");
	if (f->detect_fps < 1.0f)
		f->detect_fps = 1.0f;

	// analysis_method removed

	f->freeze_transform = obs_data_get_bool(settings, "freeze_transform");
	f->apply_translation = obs_data_get_bool(settings, "apply_translation");
	f->apply_scale = obs_data_get_bool(settings, "apply_scale");
	f->meas_alpha = (float)obs_data_get_double(settings, "meas_alpha");
	if (f->meas_alpha < 0.0f)
		f->meas_alpha = 0.0f;
	if (f->meas_alpha > 1.0f)
		f->meas_alpha = 1.0f;
	f->ease_tau = (float)obs_data_get_double(settings, "ease_tau");
	if (f->ease_tau < 0.001f)
		f->ease_tau = 0.001f;
	f->jump_guard = obs_data_get_bool(settings, "jump_guard");
	f->max_pos_jump_px = (float)obs_data_get_double(settings, "max_pos_jump_px");
	if (f->max_pos_jump_px < 0.0f)
		f->max_pos_jump_px = 0.0f;
	f->max_scale_jump = (float)obs_data_get_double(settings, "max_scale_jump");
	if (f->max_scale_jump < 0.0f)
		f->max_scale_jump = 0.0f;

	// (release) debug overlay settings removed

	// Apply detector params (worker thread also uses td, so guard with td_mutex)
	pthread_mutex_lock(&f->td_mutex);
	if (f->td) {
		// If we're doing our own external downscale, disable apriltag internal decimation.
		f->td->quad_decimate = (f->downscale_mode == 0) ? 1.0f : clampf(f->quad_decimate, 1.0f, 6.0f);
		f->td->quad_sigma = clampf(f->quad_sigma, 0.0f, 2.0f);
		f->td->refine_edges = f->refine_edges ? 1 : 0;
		f->td->decode_sharpening = clampf(f->decode_sharpening, 0.0f, 1.0f);
		f->td->qtp.deglitch = f->deglitch ? 1 : 0;
		f->td->qtp.min_cluster_pixels = f->min_cluster_pixels;
		f->td->qtp.max_nmaxima = f->max_nmaxima;
		f->td->qtp.critical_rad = f->critical_rad;
		f->td->qtp.cos_critical_rad = cosf(f->critical_rad);
		f->td->qtp.max_line_fit_mse = f->max_line_fit_mse;
		f->td->qtp.min_white_black_diff = f->min_white_black_diff;
	}
	pthread_mutex_unlock(&f->td_mutex);

	// easing tau was hardcoded; now exposed.

	f->clamp_to_canvas = obs_data_get_bool(settings, "clamp_to_canvas");
	f->clamp_margin_px = (float)obs_data_get_double(settings, "clamp_margin_px");
	if (f->clamp_margin_px < 0.0f)
		f->clamp_margin_px = 0.0f;

	// (release) manual/ROI debug transforms removed

	f->_transform_dirty = true;
}

static bool find_item_cb(obs_scene_t *scene, obs_sceneitem_t *item, void *param)
{
	UNUSED_PARAMETER(scene);
	struct {
		obs_source_t *parent;
		obs_sceneitem_t *found;
	} *ctx = param;

	if (!ctx->found && obs_sceneitem_get_source(item) == ctx->parent)
		ctx->found = item;
	return !ctx->found; // continue until found
}

static void apply_transform_task(void *param)
{
	struct trackerzoomer_filter *f = param;
	if (!f)
		return;

	if (!f->enable_tracking)
		return;

	obs_source_t *parent = NULL;
	obs_source_t *scene_source = NULL;
	obs_scene_t *scene = NULL;
	obs_sceneitem_t *item = NULL;

	if (f->parent_weak)
		parent = obs_weak_source_get_source(f->parent_weak);
	if (!parent)
		goto cleanup;

#if defined(HAVE_OBS_FRONTEND_API)
	scene_source = obs_frontend_get_current_scene();
	if (!scene_source)
		goto cleanup;
#else
	// Frontend API not available (headless build / CI configuration).
	// We can’t locate the current scene item, so just skip applying transforms.
	goto cleanup;
#endif

	scene = obs_scene_from_source(scene_source);
	if (!scene)
		goto cleanup;

	struct {
		obs_source_t *parent;
		obs_sceneitem_t *found;
	} ctx = {.parent = parent, .found = NULL};

	obs_scene_enum_items(scene, find_item_cb, &ctx);
	item = ctx.found;
	if (item)
		obs_sceneitem_addref(item);

	if (item) {
		// Apply a coherent snapshot captured on the tick thread.
		float want_x = 0.0f, want_y = 0.0f, want_s = 1.0f;
		pthread_mutex_lock(&f->xform_mutex);
		want_x = f->apply_pos_x;
		want_y = f->apply_pos_y;
		want_s = f->apply_scale_val;
		pthread_mutex_unlock(&f->xform_mutex);

		if (!f->apply_translation) {
			want_x = f->_last_applied_pos_x;
			want_y = f->_last_applied_pos_y;
		}
		if (!f->apply_scale) {
			want_s = f->_last_applied_scale;
		}

		// Only set alignment once; changing it every tick can cause unnecessary churn.
		// (OBS stores alignment in the scene item, not the source.)
		if (obs_sceneitem_get_alignment(item) != OBS_ALIGN_CENTER)
			obs_sceneitem_set_alignment(item, OBS_ALIGN_CENTER);

		struct vec2 pos = {want_x, want_y};
		struct vec2 scale = {want_s, want_s};
		obs_sceneitem_set_pos(item, &pos);
		obs_sceneitem_set_scale(item, &scale);

		f->_last_applied_pos_x = want_x;
		f->_last_applied_pos_y = want_y;
		f->_last_applied_scale = want_s;
	}

cleanup:
	if (item)
		obs_sceneitem_release(item);
	if (scene_source)
		obs_source_release(scene_source);
	if (parent)
		obs_source_release(parent);
}

static void feed_pending_from_frame(struct trackerzoomer_filter *f, struct obs_source_frame *frame)
{
	if (!f || !frame)
		return;

	// Track basic frame characteristics for logging.
	pthread_mutex_lock(&f->frame_mutex);
	f->_last_frame_format = (int)frame->format;
	f->_last_frame_w = (int)frame->width;
	f->_last_frame_h = (int)frame->height;
	pthread_mutex_unlock(&f->frame_mutex);

	const int src_w = (int)frame->width;
	const int src_h = (int)frame->height;
	if (src_w <= 0 || src_h <= 0)
		return;


	int dst_w = src_w;
	int dst_h = src_h;
	if (f->downscale_mode == 0) {
		// External downscale to a fixed working width for detection.
		dst_w = f->detect_width;
		if (dst_w < 1)
			dst_w = 1;
		if (src_w < dst_w)
			dst_w = src_w;
		dst_h = (int)((int64_t)dst_w * src_h / src_w);
		if (dst_h < 1)
			dst_h = 1;
	}

	pthread_mutex_lock(&f->frame_mutex);
	ensure_gray_frame(&f->pending, dst_w, dst_h);
	f->pending.src_w = src_w;
	f->pending.src_h = src_h;

	if (frame->format == VIDEO_FORMAT_I420 || frame->format == VIDEO_FORMAT_NV12 || frame->format == VIDEO_FORMAT_Y800 ||
	    frame->format == VIDEO_FORMAT_I444 || frame->format == VIDEO_FORMAT_I422) {
		// For planar formats (and grayscale), the first plane is luma or intensity.
		const uint8_t *yplane = frame->data[0];
		const int ystride = (int)frame->linesize[0];
		downscale_luma_nearest(f->pending.data, dst_w, dst_h, f->pending.stride, yplane, src_w, src_h, ystride);
		f->pending.frame_seq = f->_video_frame_seq;
		os_event_signal(f->worker_event);
	} else if (frame->format == VIDEO_FORMAT_UYVY) {
		const uint8_t *src = frame->data[0];
		const int stride = (int)frame->linesize[0];
		uyvy_to_gray_downscale(f->pending.data, dst_w, dst_h, f->pending.stride, src, src_w, src_h, stride);
		f->pending.frame_seq = f->_video_frame_seq;
		os_event_signal(f->worker_event);
	} else if (frame->format == VIDEO_FORMAT_YUY2) {
		const uint8_t *src = frame->data[0];
		const int stride = (int)frame->linesize[0];
		yuy2_to_gray_downscale(f->pending.data, dst_w, dst_h, f->pending.stride, src, src_w, src_h, stride);
		f->pending.frame_seq = f->_video_frame_seq;
		os_event_signal(f->worker_event);
	} else if (frame->format == VIDEO_FORMAT_YVYU) {
		const uint8_t *src = frame->data[0];
		const int stride = (int)frame->linesize[0];
		yvyu_to_gray_downscale(f->pending.data, dst_w, dst_h, f->pending.stride, src, src_w, src_h, stride);
		f->pending.frame_seq = f->_video_frame_seq;
		os_event_signal(f->worker_event);
	} else if (frame->format == VIDEO_FORMAT_BGRA) {
		const uint8_t *src = frame->data[0];
		const int stride = (int)frame->linesize[0];
		bgra_to_gray_downscale(f->pending.data, dst_w, dst_h, f->pending.stride, src, src_w, src_h, stride);
		f->pending.frame_seq = f->_video_frame_seq;
		os_event_signal(f->worker_event);
	} else if (frame->format == VIDEO_FORMAT_RGBA) {
		const uint8_t *src = frame->data[0];
		const int stride = (int)frame->linesize[0];
		rgba_to_gray_downscale(f->pending.data, dst_w, dst_h, f->pending.stride, src, src_w, src_h, stride);
		f->pending.frame_seq = f->_video_frame_seq;
		os_event_signal(f->worker_event);
	} else if (frame->format == VIDEO_FORMAT_BGRX) {
		const uint8_t *src = frame->data[0];
		const int stride = (int)frame->linesize[0];
		bgrx_to_gray_downscale(f->pending.data, dst_w, dst_h, f->pending.stride, src, src_w, src_h, stride);
		f->pending.frame_seq = f->_video_frame_seq;
		os_event_signal(f->worker_event);
	} else {
		// Unsupported format for now
	}

	// Debug: dump the downscaled grayscale image occasionally so we can verify
	// the tag is actually visible after format conversion.
	const uint64_t dump_now = os_gettime_ns();
	if ((!f->_last_dump_ns || (dump_now - f->_last_dump_ns) > 2000000000ULL) && f->_dump_count < 3) {
		dump_pgm_u8("/Users/hal9000/clawd/trackerzoomer-debug.pgm", f->pending.data, f->pending.width, f->pending.height,
		     f->pending.stride);
		blog(LOG_INFO, "[trackerzoomer-filter] wrote debug frame: /Users/hal9000/clawd/trackerzoomer-debug.pgm (%dx%d)",
		     f->pending.width, f->pending.height);
		f->_last_dump_ns = dump_now;
		f->_dump_count++;
	}

	pthread_mutex_unlock(&f->frame_mutex);
}

static void trackerzoomer_filter_tick(void *data, float seconds)
{
	UNUSED_PARAMETER(seconds);
	struct trackerzoomer_filter *f = data;
	if (!f)
		return;

	// Ensure we have a weak ref to the parent source even though we no longer
	// rely on filter_video callbacks.
	obs_source_t *parent_strong = obs_filter_get_parent(f->context);
	if (parent_strong) {
		if (!f->parent_weak)
			f->parent_weak = obs_source_get_weak_source(parent_strong);
	}

	if (!f->enable_tracking)
		return;

	// Analysis frames are obtained via filter_video callback (fixed).

	// Apply at capped rate based on OBS video FPS so easing stays smooth even if detection is sparse.
	uint64_t interval_ns = 16666666ULL;
	struct obs_video_info ovi_fps;
	if (obs_get_video_info(&ovi_fps) && ovi_fps.fps_num > 0 && ovi_fps.fps_den > 0) {
		interval_ns = (uint64_t)(1000000000ULL * (uint64_t)ovi_fps.fps_den / (uint64_t)ovi_fps.fps_num);
		// clamp to a sane range (30–240 Hz)
		if (interval_ns < 4166666ULL)
			interval_ns = 4166666ULL;
		if (interval_ns > 33333333ULL)
			interval_ns = 33333333ULL;
	}
	const uint64_t now_ns = os_gettime_ns();
	const uint64_t last_apply_ns = f->_last_apply_ns;
	if (last_apply_ns && (now_ns - last_apply_ns) < interval_ns)
		return;

	// ROI source: tracking-derived ROI only
	float roi_cx = 0.0f;
	float roi_cy = 0.0f;
	float roi_w = 0.0f;
	float roi_h = 0.0f;
	bool use_roi = false;
	bool want_full_frame = false;

	if (f->enable_tracking) {
		// Pull raw ROI under lock, but do smoothing outside the lock.
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

		if (raw_valid) {
			// Measurement smoothing (python parity). This reduces twitch before transform easing.
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
			// No tags => show full frame.
			want_full_frame = true;
		}
	}

	// Compute target from ROI if active, else full-frame, else manual transform.
	// We compute a candidate target first, then apply a small deadband (python parity)
	// to avoid chasing tiny detection noise.
	float cand_pos_x = f->target_pos_x;
	float cand_pos_y = f->target_pos_y;
	float cand_scale = f->target_scale;
	if (want_full_frame && f->parent_weak) {
		obs_source_t *parent = obs_weak_source_get_source(f->parent_weak);
		if (parent) {
			uint32_t frame_w = obs_source_get_width(parent);
			uint32_t frame_h = obs_source_get_height(parent);
			obs_source_release(parent);

			struct obs_video_info ovi;
			if (obs_get_video_info(&ovi) && frame_w > 0 && frame_h > 0) {
				const float canvas_w = (float)ovi.base_width;
				const float canvas_h = (float)ovi.base_height;
				const float sx = canvas_w / (float)frame_w;
				const float sy = canvas_h / (float)frame_h;
				float scale = (sx < sy) ? sx : sy;
				if (scale <= 0.0f)
					scale = 1.0f;

				cand_pos_x = canvas_w * 0.5f;
				cand_pos_y = canvas_h * 0.5f;
				cand_scale = scale;
			}
		}
	} else if (use_roi && f->parent_weak) {
		obs_source_t *parent = obs_weak_source_get_source(f->parent_weak);
		if (parent) {
			uint32_t frame_w = obs_source_get_width(parent);
			uint32_t frame_h = obs_source_get_height(parent);
			obs_source_release(parent);

			struct obs_video_info ovi;
			if (obs_get_video_info(&ovi) && frame_w > 0 && frame_h > 0) {
				if (roi_w < 1.0f)
					roi_w = 1.0f;
				if (roi_h < 1.0f)
					roi_h = 1.0f;

				// Note: ROI sizing/clamping is handled earlier (python-parity) when we build the ROI.
				// Avoid imposing an extra minimum ROI here; it causes the "both tags fully in view"
				// zone as tags get close.

				const float half_w = roi_w * 0.5f;
				const float half_h = roi_h * 0.5f;
				float cx = roi_cx;
				float cy = roi_cy;
				if (cx < half_w)
					cx = half_w;
				if (cy < half_h)
					cy = half_h;
				if (cx > (float)frame_w - half_w)
					cx = (float)frame_w - half_w;
				if (cy > (float)frame_h - half_h)
					cy = (float)frame_h - half_h;

				const float canvas_w = (float)ovi.base_width;
				const float canvas_h = (float)ovi.base_height;
				const float sx = canvas_w / (roi_w > 0.001f ? roi_w : 0.001f);
				const float sy = canvas_h / (roi_h > 0.001f ? roi_h : 0.001f);
				float scale = (sx < sy) ? sx : sy;
				if (scale > 8.0f)
					scale = 8.0f;

				const float fx = (float)frame_w * 0.5f;
				const float fy = (float)frame_h * 0.5f;

				float pos_x = canvas_w * 0.5f - (cx - fx) * scale;
				float pos_y = canvas_h * 0.5f - (cy - fy) * scale;

				// Clamp translation so the scaled source still covers the canvas.
				// This prevents revealing black behind the source when ROI is near edges.
				if (f->clamp_to_canvas) {
					const float margin = f->clamp_margin_px;
					const float disp_w = (float)frame_w * scale;
					const float disp_h = (float)frame_h * scale;
					const float half_disp_w = disp_w * 0.5f;
					const float half_disp_h = disp_h * 0.5f;

					// With center alignment, pos is the center of the displayed source.
					float min_x = (canvas_w - margin) - half_disp_w;
					float max_x = margin + half_disp_w;
					float min_y = (canvas_h - margin) - half_disp_h;
					float max_y = margin + half_disp_h;

					// If the displayed source is smaller than the canvas in a dimension,
					// clamping can't prevent bars; fall back to centered.
					if (min_x > max_x) {
						min_x = max_x = canvas_w * 0.5f;
					}
					if (min_y > max_y) {
						min_y = max_y = canvas_h * 0.5f;
					}

					pos_x = clampf(pos_x, min_x, max_x);
					pos_y = clampf(pos_y, min_y, max_y);
				}

				cand_pos_x = pos_x;
				cand_pos_y = pos_y;
				cand_scale = scale;
			}
		}
	}

	if (f->freeze_transform) {
		// Still run detection + update internal ROI, but don't change the applied transform.
		f->_last_apply_ns = now_ns;
		return;
	}

	// Jump guard: clamp the candidate target step so a single bad detection can't fling the frame.
	if (f->jump_guard) {
		const float dxg = cand_pos_x - f->target_pos_x;
		const float dyg = cand_pos_y - f->target_pos_y;
		const float dsg = cand_scale - f->target_scale;
		if (f->max_pos_jump_px > 0.0f) {
			cand_pos_x = f->target_pos_x + clampf(dxg, -f->max_pos_jump_px, f->max_pos_jump_px);
			cand_pos_y = f->target_pos_y + clampf(dyg, -f->max_pos_jump_px, f->max_pos_jump_px);
		}
		if (f->max_scale_jump > 0.0f) {
			cand_scale = f->target_scale + clampf(dsg, -f->max_scale_jump, f->max_scale_jump);
		}
	}

	// Deadband on target updates (python parity). Prevents micro jitter.
	const float min_pos_delta = 2.0f;   // pixels
	const float min_scale_delta = 0.002f; // absolute scale
	const float dx_t = cand_pos_x - f->target_pos_x;
	const float dy_t = cand_pos_y - f->target_pos_y;
	const float ds_t = cand_scale - f->target_scale;
	if (fabsf(dx_t) >= min_pos_delta || fabsf(dy_t) >= min_pos_delta || fabsf(ds_t) >= min_scale_delta) {
		f->target_pos_x = cand_pos_x;
		f->target_pos_y = cand_pos_y;
		f->target_scale = cand_scale;
	}

	// Exponential smoothing towards target (cheap + stable).
	// alpha = 1 - exp(-dt/tau)
	// tau is hardcoded to 0.5s (python parity)
	const float tau = f->ease_tau;
	float alpha = 1.0f;
	if (tau > 0.0001f) {
		float dt = (float)interval_ns / 1000000000.0f;
		if (last_apply_ns) {
			dt = (float)(now_ns - last_apply_ns) / 1000000000.0f;
			if (dt < 0.0f)
				dt = 0.0f;
			if (dt > 0.5f)
				dt = 0.5f; // avoid huge jumps after stalls
		}
		alpha = 1.0f - expf(-dt / tau);
		if (alpha < 0.0f)
			alpha = 0.0f;
		if (alpha > 1.0f)
			alpha = 1.0f;
	}

	f->cur_pos_x += (f->target_pos_x - f->cur_pos_x) * alpha;
	f->cur_pos_y += (f->target_pos_y - f->cur_pos_y) * alpha;
	f->cur_scale += (f->target_scale - f->cur_scale) * alpha;

	// Capture a coherent snapshot for the UI thread apply.
	pthread_mutex_lock(&f->xform_mutex);
	f->apply_pos_x = f->cur_pos_x;
	f->apply_pos_y = f->cur_pos_y;
	f->apply_scale_val = f->cur_scale;
	pthread_mutex_unlock(&f->xform_mutex);

	const float want_x = f->cur_pos_x;
	const float want_y = f->cur_pos_y;
	const float want_s = f->cur_scale;

	// Python parity: suppress tiny updates to avoid micro-jitter.
	const float pos_thresh = 1.0f;       // pixels
	const float scale_rel_thresh = 0.01f; // 1%
	const float dx = fabsf(want_x - f->_last_applied_pos_x);
	const float dy = fabsf(want_y - f->_last_applied_pos_y);
	const float s_last = f->_last_applied_scale;
	const float ds_rel = fabsf(want_s - s_last) / (fabsf(s_last) > 1e-6f ? fabsf(s_last) : 1e-6f);
	const bool changed = f->_transform_dirty || (dx >= pos_thresh) || (dy >= pos_thresh) || (ds_rel >= scale_rel_thresh);

	// Always advance apply time for stable dt/alpha, even if we skip UI update.
	f->_last_apply_ns = now_ns;

	if (!changed)
		return;

	f->_transform_dirty = false;
	obs_queue_task(OBS_TASK_UI, apply_transform_task, f, false);
}

static struct obs_source_frame *trackerzoomer_filter_video(void *data, struct obs_source_frame *frame)
{
	// IMPORTANT: never mutate or replace the incoming frame; we only *observe*
	// it for analysis and return it untouched.
	struct trackerzoomer_filter *f = data;
	if (!f || !frame)
		return frame;

	// Cache parent weak ref.
	obs_source_t *parent = obs_filter_get_parent(f->context);
	if (parent) {
		if (!f->parent_weak)
			f->parent_weak = obs_source_get_weak_source(parent);
	}

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

	// Forensics: track when the parent source size changes or filter_begin fails.
	int pw = 0, ph = 0;
	obs_source_t *parent = obs_filter_get_parent(f->context);
	if (parent) {
		pw = (int)obs_source_get_base_width(parent);
		ph = (int)obs_source_get_base_height(parent);
	}
	const uint64_t rnow = os_gettime_ns();
	const bool size_changed = (pw != 0 && ph != 0) && (pw != f->_last_render_parent_w || ph != f->_last_render_parent_h);
	if (size_changed) {
		blog(LOG_WARNING, "[trackerzoomer-filter] render parent size changed: %dx%d -> %dx%d", f->_last_render_parent_w,
		     f->_last_render_parent_h, pw, ph);
		f->_last_render_parent_w = pw;
		f->_last_render_parent_h = ph;
		f->_last_render_log_ns = rnow;
	}

	// Minimal, safe pass-through (GPU path).
	gs_effect_t *base = obs_get_base_effect(OBS_EFFECT_DEFAULT);
	if (!base) {
		obs_source_skip_video_filter(f->context);
		return;
	}

	const bool began = obs_source_process_filter_begin(f->context, GS_RGBA, 0);
	if (began) {
		obs_source_process_filter_end(f->context, base, 0, 0);

		// (release) debug overlays removed
	} else {
		// Log begin failures at most once per second.
		if (!f->_last_render_log_ns || (rnow - f->_last_render_log_ns) > 1000000000ULL) {
			blog(LOG_WARNING,
			     "[trackerzoomer-filter] process_filter_begin failed (parent %dx%d, last frame %dx%d fmt=%d)", pw, ph,
			     f->_last_frame_w, f->_last_frame_h, f->_last_frame_format);
			f->_last_render_log_ns = rnow;
		}
		obs_source_skip_video_filter(f->context);
	}
}

static struct obs_source_info trackerzoomer_filter_info = {
	.id = "trackerzoomer_filter",
	.type = OBS_SOURCE_TYPE_FILTER,
	// Use the GPU render filter path only. This avoids async filter_video pipeline quirks
	// that can cause intermittent black/cropped frames on macOS webcam sources.
	.output_flags = 0,
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

// ─────────────────────────────────────────────────────────────────────────────
// Minimal passthrough filter for isolating render pipeline issues.
// This does NO apriltag work, NO worker thread, and NO scene-item transforms.
// If glitches persist with this filter enabled, the issue is likely unrelated
// to AprilTag detection / ROI math.

struct trackerzoomer_passthrough {
	obs_source_t *context;
};

static const char *trackerzoomer_passthrough_get_name(void *unused)
{
	UNUSED_PARAMETER(unused);
	return "TrackerZoom Passthrough";
}

static void *trackerzoomer_passthrough_create(obs_data_t *settings, obs_source_t *source)
{
	UNUSED_PARAMETER(settings);
	struct trackerzoomer_passthrough *p = bzalloc(sizeof(*p));
	p->context = source;
	return p;
}

static void trackerzoomer_passthrough_destroy(void *data)
{
	struct trackerzoomer_passthrough *p = data;
	bfree(p);
}

static void trackerzoomer_passthrough_video_render(void *data, gs_effect_t *effect)
{
	UNUSED_PARAMETER(effect);
	struct trackerzoomer_passthrough *p = data;
	if (!p)
		return;

	gs_effect_t *base = obs_get_base_effect(OBS_EFFECT_DEFAULT);
	if (!base) {
		obs_source_skip_video_filter(p->context);
		return;
	}

	if (obs_source_process_filter_begin(p->context, GS_RGBA, 0)) {
		obs_source_process_filter_end(p->context, base, 0, 0);
	} else {
		obs_source_skip_video_filter(p->context);
	}
}

static struct obs_source_info trackerzoomer_passthrough_info = {
	.id = "trackerzoomer_passthrough_filter",
	.type = OBS_SOURCE_TYPE_FILTER,
	.output_flags = 0, // use GPU render path only
	.get_name = trackerzoomer_passthrough_get_name,
	.create = trackerzoomer_passthrough_create,
	.destroy = trackerzoomer_passthrough_destroy,
	.video_render = trackerzoomer_passthrough_video_render,
};

void register_trackerzoomer_filter(void)
{
	obs_register_source(&trackerzoomer_filter_info);
	obs_register_source(&trackerzoomer_passthrough_info);
}
