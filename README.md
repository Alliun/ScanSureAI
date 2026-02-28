🏥 ScanSure AI
Confidence-Aware Medical Image Segmentation (AMD Optimized)

ScanSure AI is a medical image segmentation prototype that combines deep learning with uncertainty estimation and AMD GPU optimization.

The system performs pixel-level segmentation on medical images (MRI/CT/X-ray) and generates a confidence heatmap to support reliable clinical decision-making.

🚀 Features

Automated medical image segmentation (UNet-based)

Confidence heatmap generation using Monte Carlo Dropout

Adaptive inference pipeline

GPU acceleration (ROCm compatible)

Streamlit interactive dashboard

Inference time benchmarking

🧠 How It Works

Upload medical image

Preprocess image

UNet model generates segmentation mask

Multiple forward passes estimate prediction uncertainty

System displays:

Segmentation output

Confidence heatmap

Inference time

🏗 Tech Stack

Python

PyTorch (ROCm compatible)

UNet Architecture

Streamlit

FastAPI (optional API layer)

AMD Instinct GPU

Slingshot (for scalable deployment)

📂 Project Structure
ScanSure-AI/
│
├── unet_model.py
├── utils.py
├── app.py
├── requirements.txt
└── README.md
⚙ Installation
git clone https://github.com/your-username/ScanSure-AI.git
cd ScanSure-AI
pip install -r requirements.txt
▶ Run the Application
streamlit run app.py

Open in browser:

http://localhost:8501
⚡ AMD Optimization

The model supports ROCm-enabled PyTorch for GPU acceleration.
If GPU is available, inference runs on AMD hardware automatically.

📊 Metrics

Dice Score

IoU

Inference Time

Confidence Variance Map

📌 Note

This project is developed as a prototype for demonstrating confidence-aware medical image segmentation with HPC optimization."# ScanSureAI" 
