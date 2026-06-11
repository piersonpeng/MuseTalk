from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

import cv2
import numpy as np
import yaml


COORD_PLACEHOLDER = (0.0, 0.0, 0.0, 0.0)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    defaults = raw.get("defaults", {})
    jobs = raw.get("jobs", [])
    return [deep_merge(defaults, job) for job in jobs]


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return image


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def detect_face_with_musetalk(image_path: Path) -> list[int]:
    from musetalk.utils.preprocessing import coord_placeholder, get_landmark_and_bbox

    coords, _ = get_landmark_and_bbox([str(image_path)], 0)
    if not coords or coords[0] == coord_placeholder:
        raise RuntimeError("no_face_detected")
    return [int(v) for v in coords[0]]


def get_face_bbox(image_path: Path, image_shape: tuple[int, int, int], speaker: dict[str, Any]) -> tuple[list[int], str]:
    if "face_bbox" in speaker:
        return select_face([int(v) for v in speaker["face_bbox"]], image_shape, speaker), "config_face_bbox"
    return select_face(detect_face_with_musetalk(image_path), image_shape, speaker), "musetalk_dwpose_sfd"


def select_face(face_bbox: list[int], image_shape: tuple[int, int, int], speaker: dict[str, Any]) -> list[int]:
    mode = speaker.get("mode", "single")
    if mode == "bbox":
        return [int(v) for v in speaker["box"]]

    if mode in {"single", "region", "leftmost", "rightmost"}:
        if mode == "region":
            h, w = image_shape[:2]
            x1, y1, x2, y2 = face_bbox
            cx = ((x1 + x2) / 2.0) / w
            cy = ((y1 + y2) / 2.0) / h
            xr = speaker.get("x_range", [0.0, 1.0])
            yr = speaker.get("y_range", [0.0, 1.0])
            if not (xr[0] <= cx <= xr[1] and yr[0] <= cy <= yr[1]):
                raise RuntimeError(
                    f"target_out_of_region: center=({cx:.3f}, {cy:.3f}), "
                    f"x_range={xr}, y_range={yr}"
                )
        return face_bbox

    raise ValueError(f"Unsupported speaker mode: {mode}")


def compute_square_crop(
    face_bbox: list[int],
    image_shape: tuple[int, int, int],
    crop_cfg: dict[str, Any],
) -> list[int]:
    h, w = image_shape[:2]
    x1, y1, x2, y2 = face_bbox
    face_w = max(1, x2 - x1)
    face_h = max(1, y2 - y1)

    min_face_width = int(crop_cfg.get("min_face_width_px", 0))
    if face_w < min_face_width:
        raise RuntimeError(f"face_too_small: width={face_w}, min={min_face_width}")

    expansion = float(crop_cfg.get("expansion", 2.2))
    min_crop = int(crop_cfg.get("min_crop_size_px", 256))
    size = int(round(max(face_w, face_h) * expansion))
    size = max(size, min_crop)
    size = min(size, w, h)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0 + face_h * float(crop_cfg.get("center_y_offset_ratio", 0.10))

    left = int(round(cx - size / 2.0))
    top = int(round(cy - size / 2.0))
    left = max(0, min(left, w - size))
    top = max(0, min(top, h - size))
    return [left, top, left + size, top + size]


def bbox_in_crop(face_bbox: list[int], crop_box: list[int]) -> list[int]:
    x1, y1, x2, y2 = face_bbox
    cx1, cy1, _, _ = crop_box
    return [x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1]


def draw_debug(image: np.ndarray, face_bbox: list[int], crop_box: list[int], out_path: Path) -> None:
    debug = image.copy()
    cv2.rectangle(debug, (face_bbox[0], face_bbox[1]), (face_bbox[2], face_bbox[3]), (0, 255, 0), 3)
    cv2.rectangle(debug, (crop_box[0], crop_box[1]), (crop_box[2], crop_box[3]), (255, 0, 0), 3)
    cv2.putText(debug, "face", (face_bbox[0], max(0, face_bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(debug, "crop", (crop_box[0], max(0, crop_box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    cv2.imwrite(str(out_path), debug)


def save_crop_inputs(
    image: np.ndarray,
    crop_box: list[int],
    crop_cfg: dict[str, Any],
    job_dir: Path,
) -> tuple[Path, Path]:
    x1, y1, x2, y2 = crop_box
    crop = image[y1:y2, x1:x2]
    original_path = job_dir / "crop_original.png"
    input_path = job_dir / "crop_input.png"
    cv2.imwrite(str(original_path), crop)

    model_size = int(crop_cfg.get("model_size", 512))
    resized = cv2.resize(crop, (model_size, model_size), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(str(input_path), resized)
    return original_path, input_path


def make_lower_face_mask(
    size: int,
    face_bbox_crop: list[int],
    mask_cfg: dict[str, Any],
) -> np.ndarray:
    x1, y1, x2, y2 = face_bbox_crop
    face_w = max(1, x2 - x1)
    face_h = max(1, y2 - y1)
    center = (
        int(round((x1 + x2) / 2.0)),
        int(round(y1 + face_h * float(mask_cfg.get("mouth_center_y_ratio", 0.68)))),
    )
    axes = (
        max(8, int(round(face_w * float(mask_cfg.get("axes_x_ratio", 0.72))))),
        max(8, int(round(face_h * float(mask_cfg.get("axes_y_ratio", 0.42))))),
    )
    alpha = np.zeros((size, size), dtype=np.float32)
    cv2.ellipse(alpha, center, axes, 0, 0, 360, 1.0, -1)

    feather = int(mask_cfg.get("feather_px", 30))
    if feather > 0:
        kernel = feather * 2 + 1
        if kernel % 2 == 0:
            kernel += 1
        alpha = cv2.GaussianBlur(alpha, (kernel, kernel), 0)
    return np.clip(alpha, 0.0, 1.0)


def save_mask_debug(alpha: np.ndarray, crop_original_path: Path, out_path: Path) -> None:
    crop = cv2.imread(str(crop_original_path), cv2.IMREAD_COLOR)
    if crop is None:
        return
    heat = np.zeros_like(crop)
    heat[:, :, 2] = 255
    a = alpha[:, :, None]
    preview = (crop.astype(np.float32) * (1.0 - 0.45 * a) + heat.astype(np.float32) * (0.45 * a)).astype(np.uint8)
    cv2.imwrite(str(out_path), preview)


def write_musetalk_config(crop_input: Path, audio_path: Path, output_path: Path, job_dir: Path) -> Path:
    cfg_path = job_dir / "musetalk_inference.yaml"
    data = {
        "crop_lipsync": {
            "video_path": str(crop_input),
            "audio_path": str(audio_path),
            "result_name": output_path.name,
        }
    }
    cfg_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return cfg_path


def run_musetalk(job: dict[str, Any], crop_input: Path, audio_path: Path, job_dir: Path, timeout: int | None = None) -> Path:
    model = job.get("model", {})
    output_path = job_dir / "crop_lipsync.mp4"
    inference_cfg = write_musetalk_config(crop_input, audio_path, output_path, job_dir)
    work_dir = job_dir / "musetalk_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "scripts.inference",
        "--inference_config",
        str(inference_cfg),
        "--result_dir",
        str(work_dir),
        "--unet_model_path",
        str(model.get("unet_model_path", "models/musetalkV15/unet.pth")),
        "--unet_config",
        str(model.get("unet_config", "models/musetalkV15/musetalk.json")),
        "--whisper_dir",
        str(model.get("whisper_dir", "models/whisper")),
        "--version",
        str(model.get("version", "v15")),
        "--fps",
        str(job.get("fps", 25)),
        "--batch_size",
        str(model.get("batch_size", 8)),
    ]
    if model.get("use_float16", False):
        cmd.append("--use_float16")

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
    env.setdefault("NUMBA_CACHE_DIR", str(Path(".cache/numba").resolve()))
    subprocess.run(cmd, check=True, env=env, timeout=timeout)
    produced_path = work_dir / str(model.get("version", "v15")) / output_path.name
    if produced_path.exists():
        shutil.copyfile(produced_path, output_path)
    if not output_path.exists():
        raise RuntimeError(f"musetalk_failed: expected output not found: {output_path}")
    return output_path


def composite_video(
    source_image: np.ndarray,
    crop_video: Path,
    audio_path: Path,
    crop_box: list[int],
    alpha: np.ndarray,
    final_path: Path,
    fps: int,
) -> None:
    cap = cv2.VideoCapture(str(crop_video))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open crop video: {crop_video}")

    h, w = source_image.shape[:2]
    crop_size = crop_box[2] - crop_box[0]
    temp_video = final_path.with_name(final_path.stem + "_no_audio.mp4")
    writer = cv2.VideoWriter(str(temp_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Unable to open video writer: {temp_video}")

    alpha_3 = alpha[:, :, None].astype(np.float32)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        restored = cv2.resize(frame, (crop_size, crop_size), interpolation=cv2.INTER_LANCZOS4)
        output = source_image.copy()
        x1, y1, x2, y2 = crop_box
        base = output[y1:y2, x1:x2].astype(np.float32)
        blended = restored.astype(np.float32) * alpha_3 + base * (1.0 - alpha_3)
        output[y1:y2, x1:x2] = blended.astype(np.uint8)
        writer.write(output)

    cap.release()
    writer.release()

    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "warning",
        "-i",
        str(temp_video),
        "-i",
        str(audio_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(final_path),
    ]
    subprocess.run(cmd, check=True)


def process_job(job: dict[str, Any], prepare_only_override: bool | None, musetalk_timeout: int | None) -> dict[str, Any]:
    source_image_path = Path(job["source_image"])
    audio_path = Path(job["audio_path"])
    output_root = Path(job.get("output_root", "results/cartoon_lipsync"))
    job_dir = output_root / job["job_id"]
    job_dir.mkdir(parents=True, exist_ok=True)

    image = read_image(source_image_path)
    face_bbox, face_source = get_face_bbox(source_image_path, image.shape, job.get("speaker", {}))
    crop_box = compute_square_crop(face_bbox, image.shape, job.get("crop", {}))
    face_bbox_crop = bbox_in_crop(face_bbox, crop_box)

    debug_faces = job_dir / "debug_faces.jpg"
    draw_debug(image, face_bbox, crop_box, debug_faces)
    crop_original, crop_input = save_crop_inputs(image, crop_box, job.get("crop", {}), job_dir)

    crop_size = crop_box[2] - crop_box[0]
    alpha = make_lower_face_mask(crop_size, face_bbox_crop, job.get("mask", {}))
    alpha_path = job_dir / "mask_alpha.png"
    cv2.imwrite(str(alpha_path), (alpha * 255).astype(np.uint8))
    save_mask_debug(alpha, crop_original, job_dir / "debug_mask.jpg")

    metadata = {
        "job_id": job["job_id"],
        "source_image": str(source_image_path),
        "audio_path": str(audio_path),
        "selected_face": {
            "face_bbox": face_bbox,
            "confidence": None,
            "detector": face_source,
        },
        "pose": {
            "class": "not_estimated_mvp",
            "yaw": None,
            "pitch": None,
            "roll": None,
        },
        "crop": {
            "crop_box": crop_box,
            "original_size": crop_size,
            "model_size": int(job.get("crop", {}).get("model_size", 512)),
            "crop_original": str(crop_original),
            "crop_input": str(crop_input),
        },
        "paste_back": {
            "mask_type": job.get("mask", {}).get("type", "elliptical_lower_face"),
            "mask_alpha": str(alpha_path),
            "feather_px": int(job.get("mask", {}).get("feather_px", 30)),
        },
        "debug": {
            "debug_faces": str(debug_faces),
            "debug_mask": str(job_dir / "debug_mask.jpg"),
        },
    }

    prepare_only = job.get("prepare_only", False) if prepare_only_override is None else prepare_only_override
    if not prepare_only:
        crop_video = run_musetalk(job, crop_input, audio_path, job_dir, timeout=musetalk_timeout)
        final_path = job_dir / "final.mp4"
        composite_video(
            source_image=image,
            crop_video=crop_video,
            audio_path=audio_path,
            crop_box=crop_box,
            alpha=alpha,
            final_path=final_path,
            fps=int(job.get("fps", 25)),
        )
        metadata["outputs"] = {
            "crop_lipsync": str(crop_video),
            "final": str(final_path),
        }
    else:
        metadata["outputs"] = {
            "crop_lipsync": None,
            "final": None,
        }

    write_json(job_dir / "metadata.json", metadata)
    qa_report = {
        "status": "prepared" if prepare_only else "completed",
        "warnings": ["pose_and_drift_qa_not_enabled_in_mvp"],
    }
    write_json(job_dir / "qa_report.json", qa_report)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single-image cartoon lip-sync job.")
    parser.add_argument("--config", required=True, help="Path to a cartoon lip-sync YAML config.")
    parser.add_argument("--job-id", default=None, help="Optional job_id filter.")
    parser.add_argument("--prepare-only", action="store_true", help="Only detect/crop/mask; do not run MuseTalk.")
    parser.add_argument("--musetalk-timeout", type=int, default=None, help="Optional timeout in seconds for the MuseTalk subprocess.")
    args = parser.parse_args()

    jobs = load_config(Path(args.config))
    if args.job_id:
        jobs = [job for job in jobs if job.get("job_id") == args.job_id]
        if not jobs:
            raise SystemExit(f"No job found with job_id={args.job_id}")

    for job in jobs:
        metadata = process_job(
            job,
            prepare_only_override=True if args.prepare_only else None,
            musetalk_timeout=args.musetalk_timeout,
        )
        print(json.dumps({"job_id": metadata["job_id"], "crop": metadata["crop"], "outputs": metadata["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
