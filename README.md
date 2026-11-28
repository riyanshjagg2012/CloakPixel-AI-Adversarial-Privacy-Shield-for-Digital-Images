# CloakPixel AI

CloakPixel AI is an on-device adversarial privacy shield for digital images. It helps users reduce unauthorized facial scraping, biometric tracking, and non-consensual deepfake misuse before an image is uploaded.

## 3-Minute Pitch

Every photo published online can become a biometric training or identity-matching asset. CloakPixel gives users pixel-level control by detecting facial landmarks and applying projected gradient descent perturbations around the face. The perturbations are optimized against a local image-embedding surrogate and are evaluated with a live cosine-similarity comparison, SSIM visual-fidelity score, JPEG stress test, and local face-detector verification.

The application remains 100% local: it uses PyTorch, torchvision KeypointRCNN and ResNet models, OpenCV, and client-side browser rendering. It also strips metadata-sensitive EXIF tags and reports exactly what was removed. No image or biometric embedding is uploaded to a cloud service.

## Technical Highlights

- Landmark-guided projected gradient descent using the selected perturbation strength
- 320px landmark scaling and 12-step EOT-PGD for demonstration runs
- Resize/noise expectation-over-transformation simulation during perturbation optimization
- Local image-feature cosine comparison before and after JPEG-75 compression
- SSIM visual-fidelity audit
- Local OpenCV/Keypoint face-detector verification
- EXIF metadata audit and downloadable stripped-tag report
- Startup model warmup and one-click sample portraits

## Demo Claim

Describe CloakPixel as a local privacy experiment that applies face-region perturbations and reports local image-feature cosine similarity, SSIM, and detector results for each image. Its ResNet18 image-feature model is not a dedicated face-recognition benchmark, and measured similarity does not guarantee resistance to other recognition systems. Treat a high post-compression match as a warning.

## Run

```powershell
python app.py
```

Open `http://127.0.0.1:7862/`.