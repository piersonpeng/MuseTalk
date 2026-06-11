# Face Parsing: BiSeNet → Segformer Refactor

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the broken BiSeNet face parsing module (its `79999_iter.pth` weights are a 1.6KB Google 404 HTML page) with the jonathandinu/face-parsing Segformer that is already downloaded at `models/face-parse-bisent/`.

**Architecture:** Drop-in replacement of the BiSeNet backbone with HuggingFace's `SegformerForSemanticSegmentation` + `SegformerImageProcessor`. Keep the public `FaceParsing` API (`__init__(left_cheek_width, right_cheek_width)` + `__call__(image, size, mode)` returning a `PIL.Image`) and the entire post-processing pipeline (cone kernel, cheek mask, class-ID remapping) unchanged, since CelebAMask-HQ 19-class IDs are 1:1 identical between the two models.

**Tech Stack:** PyTorch 2.12, transformers 4.39.2, OpenCV, PIL, numpy — all already in the musetalk conda env.

---

## File structure

- **Modify** `musetalk/utils/face_parsing/__init__.py` — rewrite to use Segformer (load processor + model, run inference, post-process)
- **Delete** `musetalk/utils/face_parsing/model.py` — BiSeNet class, unused after refactor
- **Delete** `musetalk/utils/face_parsing/resnet.py` — ResNet18 backbone, unused after refactor
- **Modify** `app.py` lines 140-141 — update the required-models manifest strings (cosmetic, for the Gradio model-checklist UI; does not affect functionality)

No other files change. `blending.py`, `scripts/inference.py`, `scripts/realtime_inference.py`, and `scripts/preprocess.py` consume only the public `FaceParsing` API and need no edits.

---

## Task 1: Smoke test the Segformer model loads and runs

**Files:** none (just a one-off Python invocation)

- [ ] **Step 1: Verify Segformer loads from local model dir**

Run:
```bash
source /Users/pierson/miniconda3/etc/profile.d/conda.sh && conda activate musetalk && \
python -c "
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
m = SegformerForSemanticSegmentation.from_pretrained('./models/face-parse-bisent/')
p = SegformerImageProcessor.from_pretrained('./models/face-parse-bisent/')
print('model num_labels:', m.config.num_labels)
print('processor size:', p.size)
"
```

Expected output:
```
model num_labels: 19
processor size: {'height': 512, 'width': 512}
```

- [ ] **Step 2: Verify a forward pass works on a synthetic image**

Run:
```bash
source /Users/pierson/miniconda3/etc/profile.d/conda.sh && conda activate musetalk && \
python -c "
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import numpy as np
m = SegformerForSemanticSegmentation.from_pretrained('./models/face-parse-bisent/').eval()
p = SegformerImageProcessor.from_pretrained('./models/face-parse-bisent/')
img = Image.fromarray((np.random.rand(512, 512, 3) * 255).astype('uint8'))
with torch.no_grad():
    inputs = p(images=img, return_tensors='pt')
    out = m(**inputs)
    logits = out.logits
    print('logits shape:', tuple(logits.shape))
    # upsampled to 512x512
    import torch.nn.functional as F
    up = F.interpolate(logits, size=(512, 512), mode='bilinear', align_corners=False)
    parsing = up.argmax(dim=1)[0].numpy()
    unique, counts = np.unique(parsing, return_counts=True)
    print('unique class IDs seen:', unique.tolist())
    print('pixel counts per class:', dict(zip(unique.tolist(), counts.tolist())))
"
```

Expected:
- `logits shape: (1, 19, 128, 128)` (H/4, W/4)
- A range of class IDs present (0-18), with class 0 (background) likely dominant in random noise

- [ ] **Step 3: Commit smoke-test verification (no code change, just a log in commit body)**

If this is a fresh branch, no commit needed. Otherwise skip.

---

## Task 2: Rewrite `musetalk/utils/face_parsing/__init__.py` to use Segformer

**Files:**
- Modify: `musetalk/utils/face_parsing/__init__.py` (full rewrite, ~115 lines)

- [ ] **Step 1: Write the new `__init__.py`**

The file is rewritten in full. The public API (`__init__(left_cheek_width, right_cheek_width)` and `__call__(image, size, mode)`) stays exactly the same. The model-load, preprocess, and forward-pass internals are swapped. The post-processing (`mode == "neck" / "jaw" / "raw"` branches) and the cone kernel / cheek mask setup are **unchanged** — they operate on `parsing == 1` (skin), which has identical semantics in Segformer.

Replace the entire file with:

```python
import torch
import os
import cv2
import numpy as np
from PIL import Image
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


class FaceParsing():
    def __init__(self, left_cheek_width=80, right_cheek_width=80,
                 model_path='./models/face-parse-bisent/'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = SegformerImageProcessor.from_pretrained(model_path)
        self.net = SegformerForSemanticSegmentation.from_pretrained(model_path)
        self.net.to(self.device)
        self.net.eval()

        # Build the cone kernel used to dilate the skin mask in `jaw` mode.
        cone_height = 21
        tail_height = 12
        total_size = cone_height + tail_height
        kernel = np.zeros((total_size, total_size), dtype=np.uint8)
        center_x = total_size // 2
        for row in range(cone_height):
            if row < cone_height // 2:
                continue
            width = int(2 * (row - cone_height // 2) + 1)
            start = int(center_x - (width // 2))
            end = int(center_x + (width // 2) + 1)
            kernel[row, start:end] = 1
        if cone_height > 0:
            base_width = int(kernel[cone_height - 1].sum())
        else:
            base_width = 1
        for row in range(cone_height, total_size):
            start = max(0, int(center_x - (base_width // 2)))
            end = min(total_size, int(center_x + (base_width // 2) + 1))
            kernel[row, start:end] = 1
        self.kernel = kernel

        # Flat ellipse used to erode the dilated skin mask (preserves cheek width).
        self.cheek_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 3))

        # Two vertical strips; protects the chin/center column from aggressive erosion.
        self.cheek_mask = self._create_cheek_mask(
            left_cheek_width=left_cheek_width,
            right_cheek_width=right_cheek_width,
        )

    def _create_cheek_mask(self, left_cheek_width=80, right_cheek_width=80):
        mask = np.zeros((512, 512), dtype=np.uint8)
        center = 512 // 2
        cv2.rectangle(mask, (0, 0), (center - left_cheek_width, 512), 255, -1)
        cv2.rectangle(mask, (center + right_cheek_width, 0), (512, 512), 255, -1)
        return mask

    def __call__(self, image, size=(512, 512), mode="raw"):
        if isinstance(image, str):
            image = Image.open(image)

        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.net(**inputs)
            logits = outputs.logits
            target_h, target_w = inputs["pixel_values"].shape[-2:]
            upsampled = F.interpolate(
                logits, size=(target_h, target_w), mode="bilinear", align_corners=False
            )
            parsing = upsampled.argmax(dim=1)[0].cpu().numpy()

        # Post-processing on the class-ID map. Class IDs are 1:1 identical
        # between the old BiSeNet weights and jonathandinu's Segformer
        # (both trained on CelebAMask-HQ, 19 classes, same id2label mapping).
        if mode == "neck":
            parsing[np.isin(parsing, [1, 11, 12, 13, 14])] = 255
            parsing[np.where(parsing != 255)] = 0
        elif mode == "jaw":
            face_region = np.isin(parsing, [1]) * 255
            face_region = face_region.astype(np.uint8)
            original_dilated = cv2.dilate(face_region, self.kernel, iterations=1)
            eroded = cv2.erode(original_dilated, self.cheek_kernel, iterations=2)
            face_region = cv2.bitwise_and(eroded, self.cheek_mask)
            face_region = cv2.bitwise_or(
                face_region, cv2.bitwise_and(original_dilated, ~self.cheek_mask)
            )
            parsing[(face_region == 255) & (~np.isin(parsing, [10]))] = 255
            parsing[np.isin(parsing, [11, 12, 13])] = 255
            parsing[np.where(parsing != 255)] = 0
        else:  # "raw"
            parsing[np.isin(parsing, [1, 11, 12, 13])] = 255
            parsing[np.where(parsing != 255)] = 0

        return Image.fromarray(parsing.astype(np.uint8))
```

- [ ] **Step 2: Verify the rewritten module imports and instantiates**

Run:
```bash
source /Users/pierson/miniconda3/etc/profile.d/conda.sh && conda activate musetalk && \
python -c "
from musetalk.utils.face_parsing import FaceParsing
import numpy as np
from PIL import Image
fp = FaceParsing()
img = Image.fromarray((np.random.rand(512, 512, 3) * 255).astype('uint8'))
for mode in ('raw', 'neck', 'jaw'):
    out = fp(img, mode=mode)
    arr = np.array(out)
    print(f'mode={mode}: shape={arr.shape}, dtype={arr.dtype}, unique={np.unique(arr).tolist()}')
"
```

Expected:
- No error during import or model load
- For each mode, output is a `(512, 512)` uint8 array with values in `{0, 255}` (it's a 0/255 mask, but the test just needs to confirm the post-processing didn't crash and the dtype is uint8)

- [ ] **Step 3: Verify the full inference script gets past `FaceParsing()` instantiation**

Run:
```bash
source /Users/pierson/miniconda3/etc/profile.d/conda.sh && conda activate musetalk && \
timeout 60 python -m scripts.inference --inference_config configs/inference/test.yaml 2>&1 | head -40
```

Expected: The `FaceParsing(...)` line no longer raises. The script should now either:
- Run further (and possibly hit a downstream error in audio/landmark extraction, which is expected and means our refactor is no longer the bottleneck), or
- Run to completion

Either outcome means Task 2 is successful. If the FaceParsing import or instantiation itself errors, debug the new `__init__.py` (most likely culprits: model path, processor arg name, device mismatch).

- [ ] **Step 4: Commit**

```bash
git add musetalk/utils/face_parsing/__init__.py
git commit -m "refactor(face_parsing): replace BiSeNet with HuggingFace Segformer

The BiSeNet 79999_iter.pth weights URL is dead (downloads a 1.6KB Google
404 page). The face-parse-bisent/ directory already contains
jonathandinu/face-parsing (Segformer, mit-b5 backbone, 19 classes on
CelebAMask-HQ) — swap to it.

Class ID semantics are 1:1 identical between the two models, so the
post-processing pipeline (cone kernel, cheek mask, mode branching) is
ported verbatim. The public FaceParsing API is unchanged, so blending.py,
inference.py, realtime_inference.py, and app.py need no edits.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 3: Delete the now-unused BiSeNet files

**Files:**
- Delete: `musetalk/utils/face_parsing/model.py`
- Delete: `musetalk/utils/face_parsing/resnet.py`

- [ ] **Step 1: Confirm nothing else imports from them**

Run:
```bash
grep -rn "from .model\|from .resnet\|from musetalk.utils.face_parsing.model\|from musetalk.utils.face_parsing.resnet" musetalk/ scripts/ app.py 2>/dev/null
```

Expected: **No output.** If anything is found, fix it before deleting.

- [ ] **Step 2: Delete the files**

```bash
git rm musetalk/utils/face_parsing/model.py musetalk/utils/face_parsing/resnet.py
```

- [ ] **Step 3: Verify the import chain still works**

Run:
```bash
source /Users/pierson/miniconda3/etc/profile.d/conda.sh && conda activate musetalk && \
python -c "from musetalk.utils.face_parsing import FaceParsing; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git commit -m "chore(face_parsing): remove unused BiSeNet model files

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 4: Update `app.py` model checklist strings

**Files:**
- Modify: `app.py:140-141` (the required-models manifest)

- [ ] **Step 1: Read the current state**

```bash
sed -n '135,150p' app.py
```

Look for the BiSeNet filenames in the manifest.

- [ ] **Step 2: Replace the strings**

The Segformer model uses `config.json` and `model.safetensors` (or `pytorch_model.bin`). Update both the BiSeNet weight references to the new ones, AND update the path keys if they were `face-parse-bisent/79999_iter.pth` style — leave the path keys pointing at `face-parse-bisent/` since that is the local model directory.

The exact old/new text depends on what the file looks like — read it first, then apply a focused Edit. The intent: the manifest should list `config.json`, `model.safetensors`, `preprocessor_config.json` (the Segformer essentials) instead of `79999_iter.pth` and `resnet18-5c106cde.pth`.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "chore(app): update required-models manifest for Segformer

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 5: Final end-to-end verification

**Files:** none

- [ ] **Step 1: Run the actual inference command**

```bash
source /Users/pierson/miniconda3/etc/profile.d/conda.sh && conda activate musetalk && \
python -m scripts.inference --inference_config configs/inference/test.yaml 2>&1 | tail -30
```

Expected: the script runs to completion (or fails on a non-face-parsing issue, e.g., the input video doesn't exist, or a downstream module has its own issue). The point is: the `FaceParsing(...)` instantiation is no longer a blocker.

- [ ] **Step 2: (Optional) Clean up the dead weight files in `models/face-parse-bisent/`**

```bash
rm -f models/face-parse-bisent/79999_iter.pth models/face-parse-bisent/resnet18-5c106cde.pth
```

The 1.6KB `79999_iter.pth` (404 page) and 46MB `resnet18-5c106cde.pth` (ResNet18 backbone) are not used by the Segformer model. Removing them saves ~46MB. Skip this step if you want to keep them around for any reason.

---

## Self-review

- **Spec coverage:** Task 1 verifies model loads, Task 2 does the rewrite, Task 3 cleans up dead code, Task 4 fixes the UI manifest, Task 5 does end-to-end verification. All 5 of the user's stated goals are covered.
- **Placeholder scan:** No "TBD" / "TODO" / "implement later". The post-processing code is reproduced verbatim from the original file. The "Step 2 of Task 4" is intentionally a read-then-edit because the exact manifest format may have changed since the investigation.
- **Type consistency:** `FaceParsing.__init__` accepts `left_cheek_width` and `right_cheek_width` (same defaults as before: `80`), and the new optional `model_path` keyword arg defaults to the local Segformer dir. `__call__` keeps the same `size=(512, 512)` and `mode="raw"` defaults.
- **Public API:** `__init__(left_cheek_width=80, right_cheek_width=80)` — same as before. `__call__(image, size=(512, 512), mode="raw")` — same as before. Returns `PIL.Image` — same as before. No caller (`blending.py`, `inference.py`, `realtime_inference.py`, `app.py`) needs edits.
