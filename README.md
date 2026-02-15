# TrackerZoom Filter (OBS Plugin)

AprilTag-driven axis-aligned zoom **as an OBS video filter**.

TrackerZoom uses **two AprilTags** (family `tag16h5`, IDs `0` and `1` by default) placed on a tabletop. The filter crops and scales the source so your camera centers on the midpoint between tags and zooms so the tags sit just out of shot.

- **No perspective correction** (no deskew): the crop is axis-aligned to your OBS canvas/output aspect.
- When tracking is **off**, the filter is a simple passthrough.

## Downloads

- **Releases (recommended):** https://github.com/1030/trackerzoomermac/releases
- **CI artifacts (latest builds):** https://github.com/1030/trackerzoomermac/actions

## Install

### Windows (portable OBS or standard install)

Download the latest **Windows x64** build.

Copy these files into your OBS folder:

- `obs-plugins/64bit/trackerzoomer-filter.dll`
- `data/obs-plugins/trackerzoomer-filter/effects/trackerzoomer.effect`
- `data/obs-plugins/trackerzoomer-filter/locale/en-US.ini`

For **portable OBS**, those paths are relative to the portable OBS directory.

### macOS

Download the `.pkg` and run it.

### Linux (Ubuntu)

Download the `.deb` and install it.

## Use

1. In OBS, select your source (e.g. a webcam).
2. Add a filter: **Filters → Effect Filters → + → TrackerZoom Filter**.
3. Print or display two tags (see **Tag setup** below).
4. In the filter properties:
   - Enable **AprilTag tracking**
   - Set **Tag ID A** / **Tag ID B** (defaults are `0` and `1`)

## Tag setup

- Family: `tag16h5`
- IDs: `0` and `1` (defaults)
- Place anywhere on the tabletop; rotate as you like.
- The crop is axis-aligned to your output aspect; no deskew is performed.

## Settings (quick guide)

- **Padding (px)**: Insets the crop (per side) so the tags are pushed further out of shot.
- **Min decision margin**: Confidence threshold. Increase to reduce false detections; decrease if detection is missing.
- **Max hamming**: Error correction tolerance. `0` is strict; `1`/`2` can help with noisy images.
- **Detection width (px)**: Internal downscale width for detection. Lower = faster; higher = more reliable detection.
- **Detection FPS**: How often detection runs.
- **quad_sigma / refine_edges / decode_sharpening**: AprilTag detector tuning knobs.
- **ROI smoothing (alpha)**: Smooths the measured ROI. Higher = more responsive; lower = smoother.

## Troubleshooting

- Ensure tags are large enough in frame, sharp, and well-lit.
- If detection is flaky:
  - increase **Detection width**
  - lower **Min decision margin** slightly
  - increase lighting / reduce motion blur

## Build from source

This repo uses CMake and vendors AprilTag in `third_party/apriltag`.

CI builds for Windows/macOS/Linux via GitHub Actions.

## Credits

- AprilTag detector: https://github.com/AprilRobotics/apriltag
- Inspired by TrackerZoom / TrackerZoomOBS.

## License

See `LICENSE`.
