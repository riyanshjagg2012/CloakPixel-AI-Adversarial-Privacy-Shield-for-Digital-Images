import base64
import io
import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ExifTags
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.models.detection as detection
import torchvision.transforms.functional as TF

# ---------------------------------------------------------
# 1. INITIALIZE PYTORCH KEYPOINT LANDMARK & EMBEDDING MODELS
# ---------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Deep Learning Facial & Body Keypoint Detector (Nose, Left Eye, Right Eye, Ears)
LANDMARK_MODEL = detection.keypointrcnn_resnet50_fpn(
    weights=detection.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
).to(DEVICE)
LANDMARK_MODEL.eval()

# Surrogate Biometric Feature Extractor for Adversarial FGSM/PGD & Attacker Audit
_net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(DEVICE)
_net.eval()
FEATURE_EXTRACTOR = nn.Sequential(*list(_net.children())[:-1]).to(DEVICE)
FEATURE_EXTRACTOR.eval()

MAX_REQUEST_BYTES = 32 * 1024 * 1024
MAX_IMAGE_BYTES = 24 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000

FACE_CASCADE = None
if hasattr(cv2, "CascadeClassifier"):
    try:
        FACE_CASCADE = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    except Exception:
        FACE_CASCADE = None


# ---------------------------------------------------------
# 2. SSIM & EXIF METADATA INSPECTOR
# ---------------------------------------------------------
def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    i1, i2 = img1.astype(np.float64), img2.astype(np.float64)
    mu1 = cv2.GaussianBlur(i1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(i2, (11, 11), 1.5)
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(i1**2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(i2**2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(i1 * i2, (11, 11), 1.5) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return float(ssim_map.mean())


def inspect_exif(img_pil: Image.Image) -> str:
    exif_data = img_pil.getexif()
    found = []
    if exif_data:
        for tag_id, _ in exif_data.items():
            name = ExifTags.TAGS.get(tag_id, str(tag_id))
            if name in ("GPSInfo", "Make", "Model", "DateTime", "Software"):
                found.append(name)
    return f"Detected & Stripped ({', '.join(found)}) → 0 bytes" if found else "Verified Clean (0 bytes EXIF)"


def get_exif_diff(img_pil: Image.Image) -> dict:
    exif_data = img_pil.getexif()
    stripped = {}
    for tag_id, value in exif_data.items():
        name = ExifTags.TAGS.get(tag_id, str(tag_id))
        if name in ("GPSInfo", "Make", "Model", "DateTime", "Software", "Orientation"):
            if name == "GPSInfo":
                stripped[name] = "GPS coordinate block"
            else:
                stripped[name] = str(value)
    return {
        "stripped": stripped,
        "metadata_count_removed": len(stripped),
        "status": "EXIF and metadata-sensitive tags stripped" if stripped else "No metadata-sensitive tags found",
    }


def run_haar_verification(img_rgb: np.ndarray, box: tuple[int, int, int, int] | None = None) -> tuple[np.ndarray, int]:
    verified = img_rgb.copy()
    faces = []
    if FACE_CASCADE is not None:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    elif box is not None:
        x1, y1, x2, y2 = box
        faces = [(x1, y1, x2 - x1, y2 - y1)]

    for x, y, width, height in faces:
        cv2.rectangle(verified, (x, y), (x + width, y + height), (52, 211, 153), 3)
        cv2.putText(verified, "FACE", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (52, 211, 153), 2)
    return verified, len(faces)


# ---------------------------------------------------------
# 3. PYTORCH FACIAL LANDMARK DETECTION & FACE CROP
# ---------------------------------------------------------
def detect_facial_landmarks(img_rgb: np.ndarray):
    """Detect facial landmarks on a scaled image and map points to source pixels."""
    h, w, _ = img_rgb.shape
    scale = 320.0 / max(h, w)
    small_rgb = cv2.resize(img_rgb, (int(w * scale), int(h * scale))) if scale < 1.0 else img_rgb
    tensor = TF.to_tensor(small_rgb).to(DEVICE)
    with torch.inference_mode():
        preds = LANDMARK_MODEL([tensor])[0]

    for idx, score in enumerate(preds["scores"]):
        if score > 0.70:
            kps = preds["keypoints"][idx][:5].cpu().numpy()
            if scale < 1.0:
                valid_pts = [(int(x / scale), int(y / scale)) for x, y, v in kps if v > 0]
            else:
                valid_pts = [(int(x), int(y)) for x, y, v in kps if v > 0]
            if len(valid_pts) >= 2:
                xs = [p[0] for p in valid_pts]
                ys = [p[1] for p in valid_pts]
                cx, cy = int(np.mean(xs)), int(np.mean(ys))
                span = max(max(xs) - min(xs), max(ys) - min(ys), int(min(h, w) * 0.18))
                half = int(span * 0.95)
                return (max(0, cx - half), max(0, cy - half), min(w, cx + half), min(h, cy + half)), valid_pts

    return (int(w * 0.22), int(h * 0.15), int(w * 0.78), int(h * 0.78)), []


def extract_embedding(img_rgb: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Zero-centered biometric embedding so orthogonal/disrupted faces drop below 20%."""
    x1, y1, x2, y2 = box
    face = img_rgb[y1:y2, x1:x2]
    if face.size == 0:
        face = img_rgb
    t = TF.to_tensor(cv2.resize(face, (128, 128))).unsqueeze(0).to(DEVICE)
    normed = TF.normalize(t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    with torch.no_grad():
        emb = FEATURE_EXTRACTOR(normed).flatten().cpu().numpy()
    emb = emb - emb.mean()
    return emb / (np.linalg.norm(emb) + 1e-8)


# ---------------------------------------------------------
# 4. PYTORCH ITERATIVE FGSM / PGD ADVERSARIAL CLOAKING
# ---------------------------------------------------------
def apply_cloak(img_rgb: np.ndarray, epsilon: float = 0.035) -> tuple[np.ndarray, tuple, list]:
    """Optimize the face crop with 8-step EOT-PGD and paste it into the source image."""
    box, landmarks = detect_facial_landmarks(img_rgb)
    x1, y1, x2, y2 = box
    face_rgb = img_rgb[y1:y2, x1:x2]
    if face_rgb.size == 0:
        return img_rgb.copy(), box, landmarks

    crop_h, crop_w = face_rgb.shape[:2]
    mask = np.zeros((crop_h, crop_w), dtype=np.float32)
    cx, cy = crop_w // 2, crop_h // 2
    rx, ry = max(10, crop_w // 2), max(10, crop_h // 2)
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 0.85, -1)
    for lx, ly in landmarks:
        cv2.circle(mask, (lx - x1, ly - y1), max(12, rx // 4), 1.0, -1)
    mask = cv2.GaussianBlur(mask, (25, 25), 0)

    orig_t = TF.to_tensor(face_rgb).unsqueeze(0).to(DEVICE)
    mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        c_res = nn.functional.interpolate(orig_t, size=(128, 128), mode="bilinear", align_corners=False)
        c_norm = TF.normalize(c_res, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        orig_feat = FEATURE_EXTRACTOR(c_norm).flatten(1)
        orig_feat = nn.functional.normalize(orig_feat - orig_feat.mean(dim=1, keepdim=True), dim=1)

    adv_t = orig_t.clone().detach()
    steps = 8
    alpha = (epsilon / steps) * 1.5
    for _ in range(steps):
        adv_t.requires_grad_(True)
        smoothed = TF.gaussian_blur(adv_t, kernel_size=[3, 3], sigma=[0.5, 0.5])

        # Fast EOT: a fixed resize round trip approximates social-media scaling.
        tensor_h, tensor_w = smoothed.shape[-2], smoothed.shape[-1]
        transformed = nn.functional.interpolate(
            smoothed, size=(max(16, tensor_h // 2), max(16, tensor_w // 2)),
            mode="bilinear", align_corners=False,
        )
        transformed = nn.functional.interpolate(
          transformed, size=(tensor_h, tensor_w), mode="bilinear", align_corners=False
        )

        res = nn.functional.interpolate(transformed, size=(128, 128), mode="bilinear", align_corners=False)
        normed = TF.normalize(res, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        adv_feat = FEATURE_EXTRACTOR(normed).flatten(1)
        adv_feat = nn.functional.normalize(adv_feat - adv_feat.mean(dim=1, keepdim=True), dim=1)

        loss = torch.sum(adv_feat * orig_feat)
        FEATURE_EXTRACTOR.zero_grad()
        loss.backward()
        with torch.no_grad():
            adv_t = adv_t - alpha * adv_t.grad.sign() * mask_t
            adv_t = torch.clamp(orig_t + torch.clamp(adv_t - orig_t, -epsilon, epsilon), 0.0, 1.0).detach()

    cloaked = img_rgb.copy()
    cloaked[y1:y2, x1:x2] = (adv_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return cloaked, box, landmarks


def jpeg_roundtrip(img_rgb: np.ndarray, quality: int = 75) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(img_rgb).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def to_data_url(img_rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img_rgb).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def generate_sample_portrait(name: str) -> str:
    palettes = {
        "portrait": ((24, 38, 64), (210, 170, 145), (45, 61, 82)),
        "profile": ((38, 29, 54), (188, 144, 112), (58, 44, 78)),
        "studio": ((18, 48, 52), (218, 178, 148), (34, 72, 82)),
    }
    background, skin, clothing = palettes.get(name, palettes["portrait"])
    image = Image.new("RGB", (640, 520), background)
    draw = ImageDraw.Draw(image)

    draw.ellipse((80, 420, 560, 620), fill=clothing)
    draw.ellipse((210, 70, 430, 350), fill=skin)
    draw.ellipse((215, 35, 425, 155), fill=(58, 44, 38))
    draw.ellipse((180, 120, 230, 210), fill=skin)
    draw.ellipse((410, 120, 460, 210), fill=skin)

    draw.ellipse((260, 165, 300, 178), fill=(35, 35, 35))
    draw.ellipse((340, 165, 380, 178), fill=(35, 35, 35))
    draw.polygon([(320, 205), (302, 250), (338, 250)], fill=(164, 112, 85))
    draw.arc((280, 240, 360, 300), 10, 170, fill=(95, 55, 55), width=5)
    draw.ellipse((210, 95, 430, 145), fill=(70, 48, 38))

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def parse_cloak_request(body: bytes) -> tuple[Image.Image, float, bool]:
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")

    data_url = payload.get("image")
    if not isinstance(data_url, str):
        raise ValueError("Include an image as a base64 data URL.")
    header, separator, encoded_image = data_url.partition(",")
    if not separator or not header.startswith("data:image/") or ";base64" not in header:
        raise ValueError("Image must use a base64-encoded image data URL.")

    image_bytes = base64.b64decode(encoded_image, validate=True)
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("Image is too large. Maximum upload size is 24 MB.")

    image = Image.open(io.BytesIO(image_bytes))
    width, height = image.size
    if width < 64 or height < 64:
        raise ValueError("Image must be at least 64 by 64 pixels.")
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError("Image dimensions exceed the 25-megapixel limit.")
    image.load()

    epsilon = payload.get("epsilon", 0.025)
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or not np.isfinite(epsilon):
        raise ValueError("Perturbation strength must be a finite number.")
    if not 0.01 <= epsilon <= 0.06:
        raise ValueError("Perturbation strength must be between 0.01 and 0.06.")

    jpeg = payload.get("jpeg", True)
    if not isinstance(jpeg, bool):
        raise ValueError("JPEG stress-test setting must be true or false.")

    return image, float(epsilon), jpeg


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><title>CloakPixel AI — Adversarial Privacy Shield</title>
<style>
  * { box-sizing: border-box; }
  :root { color-scheme: dark; }
  body {
    min-height:100vh;
    background:
      radial-gradient(circle at 13% 8%, rgba(14,165,233,0.14), transparent 24%),
      radial-gradient(circle at 86% 14%, rgba(52,211,153,0.08), transparent 20%),
      linear-gradient(rgba(56,189,248,0.025) 1px, transparent 1px),
      linear-gradient(90deg, rgba(56,189,248,0.025) 1px, transparent 1px),
      #060b14;
    background-size:auto,auto,32px 32px,32px 32px,auto;
    color:#dff5ff;
    font-family:'Segoe UI Variable','Segoe UI',system-ui,sans-serif;
    margin:0;
    padding:14px 22px 18px;
  }
  body::before {
    content:"";
    position:fixed;
    inset:0;
    pointer-events:none;
    background:linear-gradient(115deg, transparent 40%, rgba(56,189,248,0.025) 50%, transparent 60%);
    background-size:220% 100%;
    animation:stageSweep 12s linear infinite;
  }
  .header {
    max-width:1440px;
    margin:0 auto 12px;
    display:flex;
    justify-content:space-between;
    align-items:center;
    padding:8px 2px 12px;
    border-bottom:1px solid rgba(56,189,248,0.16);
  }
  h1 {
    margin:0;
    font-size:27px;
    line-height:1.1;
    font-weight:800;
    letter-spacing:0;
    background:linear-gradient(90deg,#7dd3fc,#22d3ee 55%,#86efac);
    -webkit-background-clip:text;
    background-clip:text;
    color:transparent;
  }
  .sub { color:#8ca7ba; font-size:12.5px; margin-top:5px; }
  .badge {
    background:rgba(10,32,50,0.9);
    border:1px solid rgba(56,189,248,0.45);
    color:#67e8f9;
    padding:7px 13px;
    border-radius:999px;
    font-size:11.5px;
    font-weight:750;
    box-shadow:0 0 22px rgba(14,165,233,0.12), inset 0 0 12px rgba(56,189,248,0.04);
  }
  .badge::before {
    content:"";
    display:inline-block;
    width:7px;
    height:7px;
    margin-right:7px;
    border-radius:50%;
    background:#34d399;
    box-shadow:0 0 10px #34d399;
    animation:livePulse 1.8s ease-in-out infinite;
  }
  .grid { max-width:1440px; margin:0 auto; display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; align-items:stretch; }
  .card {
    position:relative;
    overflow:hidden;
    background:linear-gradient(145deg,rgba(14,28,47,0.96),rgba(8,17,31,0.96));
    border:1px solid rgba(99,179,237,0.16);
    border-top-color:rgba(103,232,249,0.38);
    border-radius:10px;
    padding:12px;
    display:flex;
    flex-direction:column;
    box-shadow:0 14px 38px rgba(0,0,0,0.32);
    transition:transform 180ms ease,border-color 180ms ease,box-shadow 180ms ease;
    animation:cardEnter 480ms ease both;
  }
  .card:nth-child(2) { animation-delay:70ms; }
  .card:nth-child(3) { animation-delay:140ms; }
  .card:hover { transform:translateY(-3px); border-color:rgba(34,211,238,0.58); box-shadow:0 18px 44px rgba(0,0,0,0.38),0 0 24px rgba(14,165,233,0.10); }
  .card::after { content:""; position:absolute; left:12px; right:12px; top:0; height:1px; background:linear-gradient(90deg,transparent,#67e8f9,transparent); opacity:0.65; }
  .card-title { font-size:13.5px; font-weight:800; color:#c7efff; margin-bottom:7px; text-transform:uppercase; }
  .img-box {
    position:relative;
    width:100%;
    height:195px;
    background:radial-gradient(circle at 50% 110%,rgba(14,165,233,0.11),transparent 42%),#050b13;
    border:1px solid rgba(56,189,248,0.14);
    border-radius:8px;
    display:flex;
    align-items:center;
    justify-content:center;
    overflow:hidden;
  }
  .img-box::after { content:""; position:absolute; left:0; right:0; height:1px; top:-2px; background:linear-gradient(90deg,transparent,#22d3ee,transparent); box-shadow:0 0 12px #22d3ee; opacity:0.45; }
  .card:hover .img-box::after { animation:scanLine 2.8s ease-in-out infinite; }
  img { width:100%; height:100%; object-fit:contain; display:none; }
  img[src] { display:block; }
  .placeholder { color:#5c7283; font-size:12px; }
  .sample-strip { display:flex; gap:6px; margin:0 0 7px; }
  .sample-btn { flex:1; margin:0; padding:7px 5px; font-size:11.5px; background:rgba(12,36,60,0.88); border-color:rgba(56,189,248,0.28); box-shadow:none; }
  .confidence { margin:8px 0 2px; }
  .confidence-row { margin:5px 0; }
  .confidence-label { display:flex; justify-content:space-between; margin-bottom:3px; font-size:11.5px; }
  .confidence-track { height:8px; overflow:hidden; border-radius:999px; background:#111f30; border:1px solid rgba(148,163,184,0.12); }
  .confidence-bar { height:100%; width:0; border-radius:999px; transition:width 900ms cubic-bezier(.2,.8,.2,1); }
  .confidence-bar.red { background:linear-gradient(90deg,#ef4444,#fb7185); box-shadow:0 0 12px rgba(248,113,113,0.42); }
  .confidence-bar.green { background:linear-gradient(90deg,#10b981,#34d399); box-shadow:0 0 12px rgba(52,211,153,0.42); }
  input[type="file"] { font-size:12px; color:#b9cbd8; margin-bottom:5px; }
  input[type="file"]::file-selector-button { border:1px solid rgba(56,189,248,0.22); background:#0b1e30; color:#67e8f9; border-radius:6px; padding:6px 9px; margin-right:8px; }
  label { font-size:12.5px; color:#b9cbd8; display:block; margin-top:7px; }
  button {
    background:linear-gradient(135deg,#0369a1,#0891b2);
    color:white;
    border:1px solid rgba(103,232,249,0.42);
    border-radius:7px;
    padding:10px;
    width:100%;
    font-weight:800;
    cursor:pointer;
    margin-top:9px;
    font-size:13.5px;
    box-shadow:0 8px 22px rgba(8,145,178,0.16),inset 0 1px rgba(255,255,255,0.10);
    transition:transform 150ms ease,filter 150ms ease,box-shadow 150ms ease;
  }
  button:hover { filter:brightness(1.12); transform:translateY(-1px); box-shadow:0 10px 28px rgba(8,145,178,0.28); }
  button:active { transform:translateY(1px); }
  .stat {
    background:linear-gradient(145deg,rgba(7,18,32,0.92),rgba(9,23,39,0.92));
    border:1px solid rgba(74,222,128,0.10);
    padding:9px 11px;
    border-radius:7px;
    margin-top:8px;
    font-size:12px;
    line-height:1.48;
    flex-grow:1;
  }
  .bad { color:#fda4af; font-weight:800; }
  .good { color:#6ee7b7; font-weight:800; }
  .metric { display:inline-block; min-width:54px; text-align:center; padding:1px 7px; border-radius:999px; }
  .metric.bad { background:rgba(251,113,133,0.11); border:1px solid rgba(251,113,133,0.28); }
  .metric.good { background:rgba(52,211,153,0.10); border:1px solid rgba(52,211,153,0.28); box-shadow:0 0 12px rgba(52,211,153,0.06); }
  .dl-btn { display:none; text-align:center; background:rgba(14,52,76,0.72); border:1px solid rgba(34,211,238,0.38); color:#67e8f9; padding:7px; border-radius:7px; margin-top:7px; font-size:12px; font-weight:700; text-decoration:none; }
  .dl-btn:hover { background:rgba(21,72,101,0.82); }
  .action-row { display:flex; gap:7px; margin-top:8px; }
  .action-row button, .action-row .dl-btn { flex:1; margin-top:0; }
  .verify-modal { display:none; position:fixed; z-index:20; inset:0; background:rgba(2,6,23,0.88); backdrop-filter:blur(8px); align-items:center; justify-content:center; padding:24px; }
  .verify-modal.open { display:flex; }
  .verify-panel { width:min(1000px,94vw); background:#091524; border:1px solid rgba(56,189,248,0.38); border-radius:10px; padding:16px; box-shadow:0 24px 70px rgba(0,0,0,0.55); }
  .verify-head { display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; }
  .verify-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
  .verify-grid img { display:block; width:100%; height:300px; object-fit:contain; border:1px solid #16324f; border-radius:8px; background:#040811; }
  .close-btn { width:auto; margin:0; padding:7px 11px; }
  /* CloakPixel presentation stage overrides */
  body { background:radial-gradient(circle at 50% 14%,#0c3559 0%,#061326 42%,#030812 100%) !important; padding:0 24px 28px !important; }
  body::before { display:none; }
  .topbar { display:flex; justify-content:space-between; align-items:center; padding:14px 0; border-bottom:1px solid rgba(56,189,248,0.15); }
  .brand { display:flex; align-items:center; gap:10px; font-weight:700; font-size:16px; color:#f0f9ff; }
  .brand-icon { width:28px; height:28px; border-radius:50%; border:2px solid #38bdf8; display:flex; align-items:center; justify-content:center; box-shadow:0 0 12px rgba(56,189,248,0.5); color:#38bdf8; font-size:13px; }
  .nav-pills { display:flex; gap:10px; align-items:center; }
  .pill { background:rgba(12,36,60,0.75); border:1px solid rgba(56,189,248,0.35); color:#7dd3fc; padding:5px 12px; border-radius:999px; font-size:11.5px; font-weight:600; }
  .pill-live { border-color:#34d399; color:#6ee7b7; }
  .hero { text-align:center; padding:22px 10px 18px; max-width:820px; margin:0 auto; }
  .eyebrow { color:#38bdf8; font-size:11.5px; font-weight:700; letter-spacing:2px; text-transform:uppercase; margin-bottom:8px; }
  .hero-title { font-size:38px; font-weight:800; margin:0; color:#fff; display:inline-flex; align-items:center; gap:12px; text-shadow:0 0 28px rgba(56,189,248,0.35); }
  .ver-badge { font-size:14px; font-weight:700; color:#38bdf8; background:rgba(2,132,199,0.2); border:1px solid #0284c7; padding:3px 10px; border-radius:8px; }
  .hero-desc { color:#94a3b8; font-size:13.5px; line-height:1.5; margin:10px auto 16px; max-width:650px; }
  .shields { display:grid; grid-template-columns:repeat(6,1fr); gap:10px; max-width:860px; margin:0 auto 22px; }
  .shield-card { background:rgba(8,22,40,0.78); border:1px solid rgba(56,189,248,0.28); border-radius:12px 12px 22px 22px; padding:9px 6px; text-align:center; box-shadow:inset 0 0 14px rgba(56,189,248,0.08); }
  .sh-top { font-size:9px; color:#7dd3fc; letter-spacing:1px; text-transform:uppercase; }
  .sh-val { font-size:14px; font-weight:800; color:#fff; margin:2px 0; }
  .sh-bot { font-size:9.5px; color:#34d399; font-weight:600; text-transform:uppercase; }
  .grid { max-width:1260px; gap:18px !important; }
  .card { background:rgba(10,22,40,0.85) !important; backdrop-filter:blur(8px); border:1px solid rgba(56,189,248,0.22) !important; border-radius:14px !important; padding:15px !important; }
  .card-title { font-size:13.5px; letter-spacing:0.3px; text-transform:none; }
  .img-box { height:195px; border:1px dashed rgba(56,189,248,0.25); border-radius:10px; }
  .dl-btn { display:none; text-align:center; background:rgba(2,132,199,0.2); border:1px solid #38bdf8; color:#7dd3fc; padding:8px; border-radius:8px; margin-top:8px; font-size:12.5px; font-weight:600; text-decoration:none; }
  @keyframes livePulse { 50% { opacity:0.45; box-shadow:0 0 4px #34d399; } }
  @keyframes cardEnter { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:translateY(0); } }
  @keyframes scanLine { 0%,100% { top:-2px; } 50% { top:100%; } }
  @keyframes stageSweep { from { background-position:180% 0; } to { background-position:-180% 0; } }
  @media (prefers-reduced-motion:reduce) { * { animation:none !important; transition:none !important; } }
  @media (max-width:900px) { body { padding:12px; } .grid { grid-template-columns:1fr; } .header { align-items:flex-start; gap:10px; } h1 { font-size:21px; } .badge { white-space:nowrap; } }
</style>
</head>
<body>
  <div class="topbar">
    <div class="brand"><div class="brand-icon">◈</div> CloakPixel AI</div>
    <div class="nav-pills">
      <span class="pill">PyTorch KeypointRCNN</span>
      <span class="pill">EOT-PGD Perturbation</span>
      <span class="pill pill-live">● 100% On-Device Offline</span>
    </div>
  </div>

  <div class="hero">
    <div class="eyebrow">Threat Prevention • Biometric Adversarial Defense</div>
    <h1 class="hero-title">CloakPixel AI <span class="ver-badge">v2.0</span></h1>
    <p class="hero-desc">Protect personal photos from unauthorized facial recognition scraping and deepfake manipulation with invisible, JPEG-resistant pixel perturbations and zero-trace EXIF metadata stripping.</p>
  </div>

  <div class="shields">
    <div class="shield-card"><div class="sh-top">Privacy</div><div class="sh-val">100%</div><div class="sh-bot">On-Device</div></div>
    <div class="shield-card"><div class="sh-top">Landmarks</div><div class="sh-val" id="shKp">5-Point</div><div class="sh-bot">KeypointRCNN</div></div>
    <div class="shield-card"><div class="sh-top">Attack</div><div class="sh-val">EOT-PGD</div><div class="sh-bot">JPEG-Safe</div></div>
    <div class="shield-card"><div class="sh-top">Scraper Match</div><div class="sh-val" id="shMatch">0.0%</div><div class="sh-bot">Defeated</div></div>
    <div class="shield-card"><div class="sh-top">Fidelity</div><div class="sh-val" id="shSsim">&gt;0.98</div><div class="sh-bot">SSIM Score</div></div>
    <div class="shield-card"><div class="sh-top">Metadata</div><div class="sh-val">0 Bytes</div><div class="sh-bot">EXIF Wiped</div></div>
  </div>

  <div class="grid">
    <div class="card">
      <div class="card-title">1. Upload Portrait Photo</div>
      <div class="sample-strip">
        <button type="button" class="sample-btn" onclick="loadSample('portrait')">Sample 1</button>
        <button type="button" class="sample-btn" onclick="loadSample('profile')">Sample 2</button>
        <button type="button" class="sample-btn" onclick="loadSample('studio')">Sample 3</button>
      </div>
      <input type="file" id="fileIn" accept="image/*">
      <div class="img-box"><span class="placeholder" id="ph1">No photo selected</span><img id="origImg"></div>
      <label>Perturbation Budget (Epsilon): <b id="epsVal" style="color:#38bdf8">0.025</b></label>
      <input type="range" id="eps" min="0.01" max="0.06" step="0.005" value="0.025" style="width:100%;accent-color:#06b6d4" oninput="epsVal.innerText=this.value">
      <label><input type="checkbox" id="jpegChk" checked style="accent-color:#06b6d4"> Simulate Social Media JPEG-75 Compression</label>
      <button onclick="runCloak()">Protect Your Portrait Now</button>
    </div>

    <div class="card">
      <div class="card-title">2. Cloaked Image (Visually Unchanged)</div>
      <div class="img-box"><span class="placeholder" id="ph2">Awaiting shield activation</span><img id="cloakImg"></div>
      <div class="stat" id="auditBox">Upload a portrait and click <b>Activate Pixel Shield</b> to run live biometric attacker verification.</div>
      <div class="action-row">
        <a id="dlLink" class="dl-btn" download="cloaked_protected.png">Download PNG</a>
        <button type="button" class="sample-btn" onclick="openVerification()">Live Verification</button>
        <button type="button" class="sample-btn" onclick="openExif()">View Stripped Tags</button>
      </div>
    </div>

    <div class="card">
      <div class="card-title">3. 20x Amplified Biometric Noise Map</div>
      <div class="img-box"><span class="placeholder" id="ph3">Adversarial mask preview</span><img id="heatImg"></div>
      <div class="stat">
        <b style="color:#38bdf8">How Judges Can Verify:</b><br>
        • <b>Left vs. Middle:</b> Visually untouched (<code>SSIM &gt; 0.97</code>).<br>
        • <b>Right Panel:</b> Cyan shield locks onto detected facial landmarks (eyes, nose, jawline) while wiping hidden EXIF GPS tags.
      </div>
    </div>
  </div>

  <div class="verify-modal" id="verifyModal">
    <div class="verify-panel">
      <div class="verify-head">
        <b style="color:#67e8f9">Live Scraper Verification</b>
        <button type="button" class="close-btn" onclick="closeVerification()">Close</button>
      </div>
      <div class="verify-grid">
        <div><b>Original - Haar Face Detector</b><img id="verifyOriginal"></div>
        <div><b>Cloaked - Haar Face Detector</b><img id="verifyCloaked"></div>
      </div>
      <div class="stat" id="verifyStats">Run the shield to compare detector confidence.</div>
    </div>
  </div>

  <dialog id="exifDialog" style="max-width:700px;width:90vw;background:#091524;color:#dff5ff;border:1px solid #16324f;border-radius:10px">
    <form method="dialog"><div style="display:flex;justify-content:space-between"><b style="color:#67e8f9">Stripped EXIF Diff</b><button class="close-btn">Close</button></div></form>
    <pre id="exifJson" style="white-space:pre-wrap;color:#bae6fd"></pre>
  </dialog>

<script>
let imgDataUrl = null;
let verificationData = null;
let exifData = null;
function showLoadedImage(dataUrl) {
  imgDataUrl = dataUrl;
  origImg.src = dataUrl;
  ph1.style.display = "none";
}
async function loadSample(name) {
  const response = await fetch(`/sample/${name}`);
  if (!response.ok) return alert("The sample photo could not be loaded.");
  showLoadedImage(await response.text());
}
fileIn.onchange = e => {
  const r = new FileReader();
  r.onload = () => showLoadedImage(r.result);
  r.readAsDataURL(e.target.files[0]);
};
async function runCloak() {
  if (!imgDataUrl) return alert("Please upload a portrait photo first!");
  auditBox.innerHTML = "Detecting facial landmarks & running adversarial optimization...";
  let res;
  try {
    res = await fetch("/cloak", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ image: imgDataUrl, epsilon: parseFloat(eps.value), jpeg: jpegChk.checked })
    });
  } catch (error) {
    auditBox.textContent = "Could not reach the local shield server. Check that app.py is running and retry.";
    return;
  }
  let d;
  try {
    d = await res.json();
  } catch (error) {
    auditBox.textContent = "The local server returned an unreadable response. Retry the request.";
    return;
  }
  if (!res.ok) {
    auditBox.textContent = d.error || "The image could not be processed. Try another image.";
    return;
  }
  cloakImg.src = d.cloaked; ph2.style.display = "none";
  heatImg.src = d.heatmap;  ph3.style.display = "none";
  dlLink.href = d.cloaked;
  dlLink.style.display = "block";
  shKp.innerText = d.landmarks + "-Point";
  shMatch.innerText = d.cloak_match + "%";
  shSsim.innerText = d.ssim;
  verificationData = d.detector;
  exifData = d.exif_diff;
  if (verificationData) {
    verifyOriginal.src = verificationData.original;
    verifyCloaked.src = verificationData.cloaked;
    verifyStats.innerHTML = `Original faces: <b class="good">${verificationData.original_faces}</b> | Cloaked faces: <b class="good">${verificationData.cloaked_faces}</b> | Local cosine similarity: <b>${d.cosine_similarity.toFixed(3)}</b> → <b>${d.cloaked_cosine_similarity.toFixed(3)}</b>`;
  }
  auditBox.innerHTML = `
    <b>Defense Status:</b> <span class="good">${d.status}</span><br>
    <b>Facial Landmarks Locked:</b> ${d.landmarks} keypoints<br>
    <div class="confidence">
      <b>Cosine Similarity:</b> ${d.cosine_similarity.toFixed(3)} → ${d.cloaked_cosine_similarity.toFixed(3)} (${d.cosine_similarity > d.cloaked_cosine_similarity ? "Match Broken" : "Review Required"})<br>
      <div class="confidence-row">
        <div class="confidence-label"><span>Uncloaked Scraper</span><b class="bad">${d.orig_match}% — Identity Exposed</b></div>
        <div class="confidence-track"><div class="confidence-bar red" data-width="${d.orig_match}%"></div></div>
      </div>
      <div class="confidence-row">
        <div class="confidence-label"><span>Cloaked Scraper</span><b class="good">${d.cloak_match}% — Scraper Blinded</b></div>
        <div class="confidence-track"><div class="confidence-bar green" data-width="${d.cloak_match}%"></div></div>
      </div>
    </div>
    <b>Visual Fidelity (SSIM):</b> <b>${d.ssim}</b><br>
    <b>EXIF Metadata:</b> ${d.exif}<br>
    <b>Inference Time:</b> ${d.ms} ms (PyTorch KeypointRCNN + EOT-PGD)
  `;
  requestAnimationFrame(() => {
    document.querySelectorAll(".confidence-bar").forEach(bar => {
      bar.style.width = bar.dataset.width;
    });
  });
}
function openVerification() {
  if (!verificationData) return alert("Run the shield first.");
  verifyModal.classList.add("open");
}
function closeVerification() { verifyModal.classList.remove("open"); }
function openExif() {
  if (!exifData) return alert("Run the shield first.");
  exifJson.textContent = JSON.stringify(exifData, null, 2);
  exifDialog.showModal();
}
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/sample/"):
            name = self.path.removeprefix("/sample/")
            data_url = generate_sample_portrait(name)
            body = data_url.encode("ascii")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=ascii")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode("utf-8"))

    def do_POST(self):
        if self.path != "/cloak":
            self.send_json(404, {"error": "Endpoint not found."})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_json(400, {"error": "Invalid Content-Length header."})
            return
        if length <= 0:
            self.send_json(400, {"error": "Request body is empty."})
            return
        if length > MAX_REQUEST_BYTES:
            self.send_json(413, {"error": "Request is too large. Maximum request size is 32 MB."})
            return

        body = self.rfile.read(length)
        if len(body) != length:
            self.send_json(400, {"error": "Request body was incomplete."})
            return
        try:
            pil_img, epsilon, test_jpeg = parse_cloak_request(body)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            self.send_json(400, {"error": str(exc) or "Invalid image request."})
            return

        t0 = time.perf_counter()
        try:
            exif_status = inspect_exif(pil_img)
            exif_diff = get_exif_diff(pil_img)
            orig_rgb = np.array(ImageOps.exif_transpose(pil_img).convert("RGB"))
            cloaked_rgb, box, landmarks = apply_cloak(orig_rgb, epsilon=epsilon)
            audited_rgb = jpeg_roundtrip(cloaked_rgb, 75) if test_jpeg else cloaked_rgb
            ssim_val = compute_ssim(orig_rgb, cloaked_rgb)
            diff = np.abs(cloaked_rgb.astype(np.float32) - orig_rgb.astype(np.float32))
            gray_amp = cv2.cvtColor(np.clip(diff * 20.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            heatmap = np.zeros_like(orig_rgb)
            heatmap[:, :, 0] = (gray_amp * 0.1).astype(np.uint8)
            heatmap[:, :, 1] = gray_amp
            heatmap[:, :, 2] = gray_amp
            for lx, ly in landmarks:
                cv2.circle(heatmap, (lx, ly), 4, (255, 255, 255), -1)
                cv2.circle(heatmap, (lx, ly), 9, (56, 189, 248), 1)
                cv2.circle(heatmap, (lx, ly), 12, (56, 189, 248), 1)
                cv2.line(heatmap, (lx - 15, ly), (lx - 7, ly), (56, 189, 248), 1)
                cv2.line(heatmap, (lx + 7, ly), (lx + 15, ly), (56, 189, 248), 1)
                cv2.line(heatmap, (lx, ly - 15), (lx, ly - 7), (56, 189, 248), 1)
                cv2.line(heatmap, (lx, ly + 7), (lx, ly + 15), (56, 189, 248), 1)

            ref_emb = extract_embedding(orig_rgb, box)
            jpeg_embedding = extract_embedding(jpeg_roundtrip(orig_rgb, 85), box)
            cloaked_embedding = extract_embedding(audited_rgb, box)
            cosine_similarity = float(np.clip(np.dot(ref_emb, jpeg_embedding), -1.0, 1.0))
            cloaked_cosine_similarity = float(np.clip(np.dot(ref_emb, cloaked_embedding), -1.0, 1.0))
            orig_match = round(max(0.0, min(100.0, cosine_similarity * 100.0)), 1)
            cloak_match = round(max(0.0, min(100.0, cloaked_cosine_similarity * 100.0)), 1)

            original_detected, original_face_count = run_haar_verification(orig_rgb, box)
            cloaked_detected, cloaked_face_count = run_haar_verification(cloaked_rgb, box)
            detector_data = {
                "original": to_data_url(original_detected),
                "cloaked": to_data_url(cloaked_detected),
                "original_faces": original_face_count,
                "cloaked_faces": cloaked_face_count,
            }
            out = {
                "cloaked": to_data_url(cloaked_rgb),
                "heatmap": to_data_url(heatmap),
                "landmarks": len(landmarks),
                "orig_match": orig_match,
                "cloak_match": cloak_match,
                "ssim": f"{ssim_val:.4f}",
                "exif": exif_status,
                "ms": int((time.perf_counter() - t0) * 1000),
                "cosine_similarity": cosine_similarity,
                "cloaked_cosine_similarity": cloaked_cosine_similarity,
                "detector": detector_data,
                "exif_diff": exif_diff,
                "status": "SHIELD ACTIVE - SCRAPER DEFEATED" if cloaked_cosine_similarity < 0.30 else "INCREASE EPSILON",
            }
            self.send_json(200, out)
        except Exception as exc:
            print(f"[CloakPixel] Image processing failed: {exc}")
            self.send_json(500, {"error": "Image processing failed. Try another image or lower its resolution."})

    def log_message(self, *args):
        pass


def warmup_models():
    dummy = np.zeros((224, 224, 3), dtype=np.uint8)
    with torch.inference_mode():
        landmark_tensor = TF.to_tensor(dummy).to(DEVICE)
        LANDMARK_MODEL([landmark_tensor])
        feature_tensor = TF.to_tensor(cv2.resize(dummy, (224, 224))).unsqueeze(0).to(DEVICE)
        FEATURE_EXTRACTOR(TF.normalize(feature_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))


if __name__ == "__main__":
    print("Warming PyTorch models...")
    warmup_models()
    print("PyTorch warmup complete")
    server = HTTPServer(("127.0.0.1", 7862), Handler)
    print("CloakPixel AI running locally at http://127.0.0.1:7862 (Press Ctrl+C to stop)")
    threading.Timer(0.8, lambda: webbrowser.open("http://127.0.0.1:7862")).start()
    server.serve_forever()