# Cartoon Image Lip-Sync Pipeline Spec

## 1. Goal

Build a batch-friendly pipeline that takes a large illustrated/cartoon scene image, selects one target face, drives that face with an audio clip using MuseTalk, and composites the animated mouth/face region back into the original scene without obvious face distortion, drift, or square patch artifacts.

Primary target:
- 3D cartoon / semi-realistic character images.
- Single speaker per output video.
- Full-scene input images where the speaker face may be only part of the frame.
- Mild to moderate non-frontal faces, within defined pose limits.

Non-goals:
- Full profile face lip sync.
- Reconstructing hidden mouth regions.
- Large head motion generation.
- Multi-speaker simultaneous lip sync in the first version.
- Identity correction or face frontalization.

## 2. Supported Pose Range

The pipeline supports non-frontal faces only within conservative bounds.

Pose terms:
- `yaw`: left/right head turn.
- `pitch`: looking up/down.
- `roll`: in-plane head tilt.

Default thresholds:

| Pose | Auto | Strict Auto | Reject |
| --- | ---: | ---: | ---: |
| yaw | `<= 20 deg` | `20-35 deg` | `> 35 deg` |
| pitch | `<= 15 deg` | `15-25 deg` | `> 25 deg` |
| roll | `<= 15 deg` | `15-30 deg` | `> 30 deg` |

Behavior:
- `auto`: run normally.
- `strict_auto`: run with tighter masks, stronger drift checks, and lower tolerance for failed frames.
- `reject`: do not run automatically; send to manual review or regenerate the source image.

Important rule:
- Roll can be corrected by rotating the crop around the eye line and rotating back during paste-back.
- Yaw must not be "fixed" by stretching or frontalizing. Strong yaw should be rejected or downgraded.

## 3. End-to-End Flow

```text
input scene image
  -> face detection
  -> target face selection
  -> pose estimation and quality gate
  -> stable square crop computation
  -> optional roll alignment
  -> MuseTalk local crop inference
  -> generated crop video
  -> per-frame drift validation
  -> mouth/lower-face mask generation
  -> feathered paste-back onto original scene
  -> final audio/video mux
  -> metadata and QA artifacts
```

## 4. Inputs

Minimum input per job:

```yaml
job_id: ep02_int1_emma_line001
source_image: /path/to/INT1.jpg
audio_path: /path/to/line001.wav
speaker:
  mode: region
  x_range: [0.0, 0.55]
  y_range: [0.1, 0.85]
output_path: results/ep02_int1_emma_line001.mp4
```

Supported speaker selection modes:

```yaml
speaker:
  mode: leftmost
```

```yaml
speaker:
  mode: rightmost
```

```yaml
speaker:
  mode: region
  x_range: [0.0, 0.45]
  y_range: [0.15, 0.8]
```

```yaml
speaker:
  mode: bbox
  box: [x1, y1, x2, y2]
```

Future mode:

```yaml
speaker:
  mode: identity
  reference_image: /path/to/character_ref.png
```

## 5. Outputs

Each job writes:

```text
results/<job_id>/
  final.mp4
  crop_input.png
  crop_lipsync.mp4
  debug_faces.jpg
  debug_crop_box.jpg
  debug_mask_preview.mp4
  metadata.json
  qa_report.json
```

`metadata.json`:

```json
{
  "job_id": "ep02_int1_emma_line001",
  "source_image": "/path/to/INT1.jpg",
  "audio_path": "/path/to/line001.wav",
  "selected_face": {
    "face_bbox": [520, 260, 650, 410],
    "landmarks": {},
    "confidence": 0.94
  },
  "pose": {
    "yaw": 12.5,
    "pitch": 4.2,
    "roll": -6.8,
    "class": "auto"
  },
  "crop": {
    "crop_box": [455, 195, 715, 455],
    "original_size": 260,
    "model_size": 512,
    "roll_aligned": true,
    "roll_degrees": -6.8
  },
  "paste_back": {
    "mask_type": "semantic_lower_face",
    "feather_px": 28
  }
}
```

`qa_report.json`:

```json
{
  "status": "passed",
  "pose_gate": "passed",
  "face_size_gate": "passed",
  "drift_gate": "passed",
  "max_eye_drift_px": 2.1,
  "max_nose_drift_px": 3.8,
  "failed_frame_ratio": 0.0,
  "warnings": []
}
```

## 6. Face Detection And Target Selection

Detector order:
1. Existing MuseTalk face/DWPose stack if reliable for the input domain.
2. InsightFace or another robust detector if cartoon face detection is weak.
3. Manual bbox fallback from config.

All detections are normalized into:

```python
FaceDetection = {
    "bbox": [x1, y1, x2, y2],
    "landmarks": {
        "left_eye": [x, y],
        "right_eye": [x, y],
        "nose": [x, y],
        "mouth_left": [x, y],
        "mouth_right": [x, y]
    },
    "score": 0.0
}
```

Target selection rules:
- Never rely on raw detector order.
- `region` filters detections by bbox center inside the configured normalized ranges.
- `leftmost` sorts by bbox center `x`.
- `rightmost` sorts by bbox center `x`.
- `bbox` selects the detection with maximum IoU against the configured box, or uses the box directly if no detector match is found.

Ambiguity handling:
- If multiple faces pass with similar scores and no clear target, fail with `ambiguous_target`.
- Save `debug_faces.jpg` with numbered boxes and centers.

## 7. Pose Estimation Gate

Pose estimation can use a 2D landmark approximation in v1, then upgrade to a 3D head pose estimator.

Required output:

```python
Pose = {
    "yaw": float,
    "pitch": float,
    "roll": float,
    "class": "auto" | "strict_auto" | "reject"
}
```

Gate policy:

```yaml
pose_policy:
  auto:
    max_yaw: 20
    max_pitch: 15
    max_roll: 15
  strict_auto:
    max_yaw: 35
    max_pitch: 25
    max_roll: 30
  on_reject: manual_review
```

For `strict_auto`:
- Use smaller paste-back mask.
- Preserve more of the original face.
- Reject if landmark drift exceeds strict thresholds.

## 8. Crop Geometry

The crop must preserve geometry. No non-uniform scaling is allowed.

Algorithm:
1. Compute face bbox center.
2. Let `face_w = x2 - x1`, `face_h = y2 - y1`.
3. Let `base = max(face_w, face_h)`.
4. Let `crop_size = base * expansion`.
5. Shift crop center slightly downward to include mouth and chin.
6. Clamp crop to image bounds.
7. Expand/pad to maintain square.
8. Resize square crop to `model_size`, usually `512`.

Default crop config:

```yaml
crop:
  model_size: 512
  expansion: 2.2
  center_y_offset_ratio: 0.12
  min_face_width_px: 120
  min_crop_size_px: 220
  max_crop_size_ratio: 0.65
```

Reject or manual-review if:
- face width is below `min_face_width_px`;
- mouth is too close to crop boundary;
- crop would include too little lower face;
- face bbox confidence is low.

## 9. Roll Alignment

If `abs(roll) > roll_align_threshold`, align the crop before MuseTalk.

Default:

```yaml
roll_alignment:
  enabled: true
  threshold_deg: 5
  max_deg: 30
```

Procedure:
1. Rotate the square crop around its center by `-roll`.
2. Run MuseTalk on the aligned crop.
3. Rotate generated frames back by `+roll`.
4. Paste back using the original unrotated crop box.

Do not use roll alignment if it causes crop padding to expose empty regions near the mouth.

## 10. MuseTalk Integration

The pipeline should call MuseTalk at the crop level.

Input:
- `crop_input.png`
- `audio_path`
- fixed MuseTalk v1.5 config

Output:
- `crop_lipsync.mp4`

Recommended implementation:
- Add a thin Python wrapper around the existing inference logic rather than shelling out for every frame.
- For MVP, shelling out to `python -m scripts.inference` is acceptable.

Important:
- The full scene image is not passed directly to MuseTalk.
- MuseTalk output is always resized back to the exact original crop size.
- The original source image remains the structural base.

## 11. Paste-Back And Feathering

Production paste-back should not replace the entire generated crop unless explicitly requested.

Preferred mask:
- semantic lower-face mask from face parsing;
- includes mouth, lips, lower cheek, and chin;
- excludes eyes, brows, hair, ears, and most nose area;
- asymmetric when the face is yawed.

Fallback mask:
- ellipse centered around mouth and lower face;
- feathered by Gaussian blur.

Default mask config:

```yaml
mask:
  type: semantic_lower_face
  fallback: elliptical_lower_face
  feather_px: 28
  strict_feather_px: 18
  include_chin: true
  include_cheeks: true
  exclude_eyes: true
  exclude_hair: true
  preserve_nose_bridge: true
```

Compositing formula:

```text
output = generated_crop * alpha + original_crop * (1 - alpha)
```

Rules:
- Alpha must be float32 in `[0, 1]`.
- Alpha dimensions must exactly match the restored crop frame.
- Composite in RGB or BGR consistently; do not mix channel order.
- Preserve original background outside `crop_box`.

## 12. Drift And Distortion QA

The pipeline must validate generated frames before final paste-back.

Reference points:
- left eye
- right eye
- nose tip
- mouth corners

For each generated frame:
1. Detect landmarks inside generated crop.
2. Compare stable landmarks against original crop landmarks.
3. Ignore mouth movement for normal lip motion, but check mouth center for extreme jumps.

Default thresholds:

```yaml
qa:
  max_eye_drift_px_auto: 5
  max_nose_drift_px_auto: 6
  max_eye_drift_px_strict: 3
  max_nose_drift_px_strict: 4
  max_failed_frame_ratio: 0.05
  on_failed_frame: use_original_crop
  on_failed_job: manual_review
```

If a frame fails:
- MVP: use the original static crop for that frame.
- Later: use previous valid frame or apply affine correction from stable landmarks.

If too many frames fail:
- mark job as failed;
- do not produce a silent bad final unless `allow_degraded_output` is true.

## 13. Batch Config

Batch file:

```yaml
defaults:
  output_root: results/cartoon_lipsync
  fps: 25
  model:
    version: v15
    use_float16: true
    unet_model_path: models/musetalkV15/unet.pth
    unet_config: models/musetalkV15/musetalk.json
  crop:
    model_size: 512
    expansion: 2.2
    center_y_offset_ratio: 0.12
  pose_policy:
    auto:
      max_yaw: 20
      max_pitch: 15
      max_roll: 15
    strict_auto:
      max_yaw: 35
      max_pitch: 25
      max_roll: 30
  mask:
    type: semantic_lower_face
    fallback: elliptical_lower_face
    feather_px: 28
  qa:
    max_failed_frame_ratio: 0.05

jobs:
  - job_id: ep02_int1_emma_line001
    source_image: /Users/pierson/project/prompt-engineering/scenes/ep02-first-night/INT1.jpg
    audio_path: data/audio/eng.wav
    speaker:
      mode: region
      x_range: [0.0, 0.55]
      y_range: [0.1, 0.85]
```

## 14. CLI Design

MVP commands:

```bash
python -m scripts.cartoon_lipsync.detect \
  --image /path/to/INT1.jpg \
  --out results/debug_detect
```

```bash
python -m scripts.cartoon_lipsync.run \
  --config configs/cartoon_lipsync/batch.yaml
```

Debug-only command:

```bash
python -m scripts.cartoon_lipsync.crop \
  --image /path/to/INT1.jpg \
  --speaker-mode region \
  --x-range 0.0 0.55 \
  --y-range 0.1 0.85 \
  --out results/crop_debug
```

## 15. Failure Codes

Standard failure codes:

```text
no_face_detected
ambiguous_target
target_out_of_region
face_too_small
pose_rejected
crop_invalid
musetalk_failed
landmark_drift_failed
mask_generation_failed
paste_back_failed
ffmpeg_failed
```

Each failure writes a `qa_report.json` and debug image if possible.

## 16. Implementation Phases

Phase 1: Geometry MVP
- Detect target face.
- Select target by region/leftmost/rightmost/bbox.
- Estimate rough roll and face size.
- Write crop image and metadata.
- Paste a static crop back with feathered ellipse mask.

Phase 2: MuseTalk Crop Inference
- Run MuseTalk on the crop image and audio.
- Restore generated crop video to original crop size.
- Composite full output video.

Phase 3: Pose And QA
- Add yaw/pitch/roll gates.
- Add landmark drift checks.
- Add debug overlays and QA reports.

Phase 4: Better Masks
- Integrate semantic lower-face mask.
- Add strict mode masks for moderate yaw.
- Add fallback elliptical lower-face mask.

Phase 5: Batch Production
- YAML batch runner.
- Resume/skip completed jobs.
- Summary CSV/JSON report.
- Manual review queue for rejected jobs.

## 17. Acceptance Criteria

A job is considered successful when:
- target face is selected deterministically;
- crop is square and restored to the exact original crop size;
- final output keeps the original scene unchanged outside the face region;
- eyes, nose, hair, and face outline do not visibly drift;
- mouth movement is visible and synchronized with audio;
- no square patch edge is visible at normal playback size;
- `qa_report.json` status is `passed` or `passed_with_warnings`.

For strict-auto pose jobs:
- visible distortion must be lower priority than preserving identity;
- if the generated mouth looks weak but the face is stable, prefer stable output over aggressive mouth replacement.

## 18. Key Design Principle

MuseTalk should contribute mouth motion, not replace the character identity.

The original image remains the source of truth for face shape, eyes, hair, lighting, and background. The generated crop is only a controlled source for mouth/lower-face motion, constrained by geometry, mask, and QA gates.

## 19. Concrete Single-Image Test

Initial test fixture:

```yaml
source_image: /Users/pierson/project/prompt-engineering/scenes/ep02-first-night/INT1.jpg
audio_path: /Users/pierson/project/prompt-engineering/scenes/ep02-first-night/audio/007_Emma.mp3
speaker:
  mode: region
  x_range: [0.20, 0.42]
  y_range: [0.20, 0.52]
```

Config file:

```text
configs/cartoon_lipsync/single_int1_emma.yaml
```

Prepare-only smoke test:

```bash
/Users/pierson/miniconda3/envs/musetalk/bin/python -m scripts.cartoon_lipsync.run_single \
  --config configs/cartoon_lipsync/single_int1_emma.yaml \
  --prepare-only
```

Full single-image test:

```bash
/Users/pierson/miniconda3/envs/musetalk/bin/python -m scripts.cartoon_lipsync.run_single \
  --config configs/cartoon_lipsync/single_int1_emma.yaml \
  --musetalk-timeout 900
```

On a CPU-only run this can be slow. The prepare-only test is the required geometry smoke test; the full test should be run on a CUDA-capable machine or with a deliberately short audio clip when validating the end-to-end composite path.

Expected artifacts:

```text
results/cartoon_lipsync/ep02_first_night_int1_007_emma/
  crop_original.png
  crop_input.png
  crop_lipsync.mp4
  debug_faces.jpg
  debug_mask.jpg
  final.mp4
  mask_alpha.png
  metadata.json
  qa_report.json
```
