# 🧠 AI vs Human Generated Image Classification  
*A Two-Stage Frequency-Based Approach for Detecting AI-Generated Images*  

### 👨‍💻 Author    
📘 [Kaggle Notebook](https://www.kaggle.com/code/bhanukacse22/ai-vs-human-generated-image-classification)  
📦 [Dataset](https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset)

---

## 🏆 Overview  
With the rise of powerful **Generative AI models** (GANs, Diffusion Models), it’s becoming increasingly difficult to tell whether an image is real or AI-generated.  
This project proposes a **two-stage frequency-based detection system** that uses both **deep learning** and **statistical analysis** to classify images as *AI-generated* or *real (human-created)*.

---

## 🧩 Key Features  

1. **Frequency-Aware CNN**  
   - Extends a pretrained ResNet18 model to process both RGB (spatial) and FFT-based (frequency) features.  
   - Learns mid-frequency patterns that often contain AI fingerprints.  

2. **Histogram-Based Refinement**  
   - Uses 3D color histograms and Kernel Density Estimation (KDE) to correct low-confidence CNN predictions.  

3. **Data Augmentation**  
   - Robust training through **JPEG compression**, **Gaussian blur**, **color jitter**, and **random flips**.  

4. **High Accuracy**  
   - Achieved **Public Leaderboard AUC = 0.9709** and **Private AUC = 0.9657** on Kaggle.  

---

## 📊 Dataset  

**Source:** [AI vs Human Generated Dataset](https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset)  
- Total images: ~80,000 (balanced dataset)  
- 39,975 real images  
- 39,975 AI-generated images (from various GANs and diffusion models)  
- Images resized to **256×256 pixels** before training  

---

## 🧠 Model Architecture  

### 🔹 Stage 1: Frequency-Aware CNN  
- Input: 6 channels (3 RGB + 3 frequency)  
- Base: Modified **ResNet18** (pretrained on ImageNet)  
- Optimizer: **Adam**, Learning rate = 3e-5  
- Loss: **Binary Cross Entropy (BCE)**  

### 🔹 Stage 2: Histogram-Based Statistical Checker  
- Computes 3D RGB histograms  
- Uses **Kernel Density Estimation (KDE)** to identify real-image color distribution  
- Corrects uncertain CNN predictions (probability ≈ 0.5)  

---

## ⚙️ Implementation Details  

| Component | Description |
|------------|-------------|
| Framework | PyTorch 2.x |
| Input Size | 256×256 |
| Batch Size | 16 |
| Augmentations | JPEG, Gaussian Blur, Color Jitter, Flip |
| Evaluation Metrics | F1, AUC-ROC, Accuracy, Confusion Matrix |
| Hardware | Kaggle GPU (T4 / V100) |
| Reproducibility | Random seed fixed for NumPy + PyTorch |

---

## 📈 Results  

| Metric | Train | Validation |
|--------|--------|-------------|
| Loss | 0.0108 | — |
| F1 Score | 0.9964 | 0.9920 |
| AUC | 0.9999 | 0.9997 |
| Accuracy | 0.9964 | 0.9920 |

- **Public AUC:** 0.9709  
- **Private AUC:** 0.9657  

---

## 🔍 Visual Insights  

- **Frequency Visualization:**  
  - Real images → smooth mid-frequency patterns  
  - AI images → grid-like frequency artifacts  

- **Histogram Analysis:**  
  - Distinct color distribution between real and AI-generated images  

---

## 🧾 References  

1. Wesselkamp et al. – *Misleading Deep-Fake Detection with GAN Fingerprints* (2022)  
2. Hendrycks & Dietterich – *Benchmarking Neural Network Robustness* (2019)  
3. Yang et al. – *Contrast Enhancement Forensics Using Pixel and Histogram CNNs* (2019)  
4. Tan et al. – *Frequency-Aware Deepfake Detection* (2024)  
5. Yu et al. – *Attributing Fake Images to GANs* (2019)  
6. Giudice et al. – *Fighting Deepfakes by Detecting GAN DCT Anomalies* (2021)  
7. Pontorno et al. – *On the Exploitation of DCT-Traces in Generative-AI* (2024)  
8. He et al. – *Deep Residual Learning for Image Recognition (ResNet)* (2016)  

---

## 🧰 How to Run  

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/AI-vs-Human-Generated-Image-Classification.git
   cd AI-vs-Human-Generated-Image-Classification
