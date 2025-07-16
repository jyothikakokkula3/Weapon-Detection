# 🔫 Weapon Detection using Deep Learning (AI)

A real-time weapon detection system built using deep learning techniques to enhance surveillance and public safety. This project uses Convolutional Neural Networks (CNNs) and the YOLO (You Only Look Once) object detection algorithm for identifying weapons in images and video streams.

---

## 📌 Features

- 🚨 Real-time detection of weapons (guns, knives, etc.)
- 🧠 Powered by YOLO and CNN-based deep learning models
- 📸 Works with both image and video input (live or pre-recorded)
- 🎯 High accuracy and precision through model optimization
- 📊 Visual bounding boxes for detected weapons

---

## 🛠 Tech Stack

- **Language:** Python
- **Libraries:** 
  - TensorFlow / PyTorch
  - OpenCV
  - NumPy
  - Matplotlib (for visualization)
- **Model:** YOLOv5 (or YOLOv4/YOLOv3 depending on use)
- **Tools:** Jupyter Notebook, Google Colab, or local IDE

---

## 📁 Folder Structure

weapon-detection/
├── data/ # Training & test images/videos
├── models/ # Trained model weights
├── notebooks/ # Jupyter Notebooks
├── utils/ # Helper functions
├── detection.py # Main script for detection
├── train.py # Model training script
├── requirements.txt # Project dependencies
└── README.md # Project documentation


---

## 🚀 Getting Started

### ✅ Prerequisites

- Python 3.8+
- pip or conda

### 📥 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/weapon-detection.git
cd weapon-detection

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

🎯 Usage
Run Detection on an Image
python detection.py --source 'image.jpg'

Run Detection on a Video
python detection.py --source 'video.mp4'

Real-Time Detection via Webcam
python detection.py --source 0

🧠 Model Training
If you wish to train the model on a custom dataset:

python train.py --data data.yaml --cfg yolov5s.yaml --weights '' --batch-size 16 --epochs 50

💡 Applications
✅ Smart surveillance systems

✅ School & public place security

✅ Law enforcement support

✅ Automated alert systems

📃 Research Background
This project explores the effectiveness of deep learning in public safety applications using object detection frameworks like YOLO. By training on a curated dataset of weapon images, it demonstrates how AI can assist in preventing crime proactively.

📄 License
This project is licensed under the MIT License.