# CloakPixel AI

CloakPixel AI is an on-device adversarial privacy shield for digital images. It helps users reduce unauthorized facial scraping, biometric tracking, and non-consensual deepfake misuse before an image is uploaded.

## 3-Minute Pitch

Every photo published online can become a biometric training or identity-matching asset. CloakPixel gives users pixel-level control by detecting facial landmarks and applying projected gradient descent perturbations around the face. The perturbations are optimized against a local image-embedding surrogate and are evaluated with a live cosine-similarity comparison, SSIM visual-fidelity score, JPEG stress test, and local face-detector verification.

The application remains 100% local: it uses PyTorch, torchvision KeypointRCNN and ResNet models, OpenCV, and client-side browser rendering. It also strips metadata-sensitive EXIF tags and reports exactly what was removed. No image or biometric embedding is uploaded to a cloud service.

## Technical Highlights

- Landmark-guided projected gradient descent on a local biometric surrogate
- 384px landmark scaling and 3-step PGD for fast demonstration runs
- Resize/noise expectation-over-transformation simulation for compression robustness
- Real cosine-similarity comparison between original and cloaked embeddings
- SSIM visual-fidelity audit
- Local OpenCV/Keypoint face-detector verification
- EXIF metadata audit and downloadable stripped-tag report
- Startup model warmup and one-click sample portraits

## Demo Claim

Say “CloakPixel adds invisible, face-region perturbations that disrupt automated embedding extraction while preserving human-visible image quality.” Use measured cosine, SSIM, and detector values as the evidence for each test image. Avoid claiming universal resistance against every future recognition or generative-AI model.

## Run

```powershell
python app.py
```

Open `http://127.0.0.1:7862/`.