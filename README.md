# 😊 Emotion Detection AI

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://12-emotion-detection-app.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Real-time facial emotion recognition using Deep Learning** | Built with PyTorch, OpenCV, and Streamlit

![Demo](https://raw.githubusercontent.com/slastrzelec/12_emotion-detection-app/main/demo_screenshot.png)

---

## 🎯 Project Overview

This project demonstrates end-to-end Machine Learning pipeline for emotion detection from facial images. The application uses a custom Convolutional Neural Network (CNN) trained on the FER-2013 dataset to classify 7 emotions in real-time.

### **Live Demo**
👉 **[Try the app here](https://12-emotion-detection-app.streamlit.app)**

---

## ✨ Features

- 🔍 **Real-time face detection** using OpenCV Haar Cascade
- 🧠 **Custom CNN model** with 4 convolutional layers
- 🎭 **7 emotion classes**: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- 📊 **Confidence scores** with probability distribution
- 🌐 **Web deployment** on Streamlit Community Cloud
- ☁️ **AWS integration** for data storage (S3)

---

## 🛠️ Tech Stack

**Deep Learning & Computer Vision:**
- PyTorch 2.10
- OpenCV (opencv-python-headless)
- torchvision

**Web Framework:**
- Streamlit

**Cloud & Storage:**
- AWS S3 (data storage)
- Streamlit Community Cloud (deployment)

**Tools:**
- Python 3.10+
- Git & GitHub
- Jupyter Notebooks

---

## 📊 Model Architecture
```
EmotionCNN(
  Input: 48x48 grayscale image
  
  Conv Block 1: Conv2D(1→64) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
  Conv Block 2: Conv2D(64→128) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
  Conv Block 3: Conv2D(128→256) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
  Conv Block 4: Conv2D(256→512) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
  
  Classifier: Flatten → Linear(4608→512) → ReLU → Dropout(0.5) → Linear(512→7)
  
  Output: 7 emotion probabilities
)
```

**Parameters:** ~11 Million  
**Training:** 20 epochs on CPU (~2.5 hours)  
**Optimizer:** Adam (lr=0.001)  
**Loss:** CrossEntropyLoss  

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 59.64% |
| **Training Images** | 28,709 |
| **Test Images** | 7,178 |
| **Inference Time** | ~50ms per face |

**Note:** 59.64% accuracy for 7 classes is solid (random baseline = 14.3%). Human agreement rate on this dataset is ~65-70%.

---

## 🚀 Quick Start

### **1. Clone the repository**
```bash
git clone https://github.com/slastrzelec/12_emotion-detection-app.git
cd 12_emotion-detection-app
```

### **2. Install dependencies**
```bash
pip install -r requirements.txt
```

### **3. Run the app locally**
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📁 Project Structure
```
12_emotion-detection-app/
├── app.py                          # Streamlit web application
├── emotion_model_best.pth          # Trained model weights (45MB)
├── test_photo.jpg                  # Example test image
├── requirements.txt                # Python dependencies
├── packages.txt                    # System dependencies
├── README.md                       # Project documentation
└── .gitignore                      # Git ignore file
```

---

## 🎓 Dataset

**FER-2013** (Facial Expression Recognition)
- **Source:** [Kaggle - FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)
- **Images:** 35,888 (48x48 grayscale)
- **Classes:** 7 emotions (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise)
- **Split:** 80% train, 20% test

---

## 💡 Key Learnings

**Technical Skills:**
- Designed and trained custom CNN from scratch
- Implemented end-to-end ML pipeline (data → training → deployment)
- Integrated OpenCV for real-time face detection
- Deployed production-ready ML app on cloud
- Optimized model for CPU inference

**ML/DL Concepts:**
- Convolutional Neural Networks (CNNs)
- Batch Normalization & Dropout for regularization
- Data augmentation techniques
- Transfer learning considerations
- Model optimization for production

**Tools & Deployment:**
- Streamlit for rapid prototyping
- AWS S3 for cloud storage
- Git for version control
- Docker considerations for containerization

---

## 🔮 Future Improvements

- [ ] Implement **transfer learning** (ResNet/VGG pretrained)
- [ ] Add **data augmentation** (rotation, brightness, contrast)
- [ ] Deploy on **AWS EC2** with GPU for faster inference
- [ ] Implement **real-time webcam** emotion detection
- [ ] Add **emotion tracking over time** (video analysis)
- [ ] Improve face detection with **MTCNN** or **RetinaFace**
- [ ] Model quantization for **edge deployment**
- [ ] A/B testing different architectures

---

## 📸 Screenshots

### Main Interface
![Main Interface](https://via.placeholder.com/800x400.png?text=Add+Your+Screenshot)

### Analysis Results
![Results](https://via.placeholder.com/800x400.png?text=Add+Your+Screenshot)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Sławomir Strzelec**  
Data Scientist | AI/ML Engineer

📧 Email: sla.strzelec@gmail.com  
🔗 LinkedIn: [linkedin.com/in/sławomir-strzelec-20a65b280](https://linkedin.com/in/sławomir-strzelec-20a65b280)  
💻 GitHub: [github.com/slastrzelec](https://github.com/slastrzelec)  
🌐 Portfolio: [Your Portfolio URL]

---

## 🙏 Acknowledgments

- **Dataset:** FER-2013 from Kaggle
- **Framework:** PyTorch Team
- **Deployment:** Streamlit Community Cloud
- **Inspiration:** Computer Vision research community

---

## 📊 Project Stats

![GitHub stars](https://img.shields.io/github/stars/slastrzelec/12_emotion-detection-app?style=social)
![GitHub forks](https://img.shields.io/github/forks/slastrzelec/12_emotion-detection-app?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/slastrzelec/12_emotion-detection-app?style=social)

---

**⭐ If you found this project useful, please consider giving it a star!**
```

---

### **2. Screenshot dla README**

Zrób screenshoty aplikacji:

1. **Main interface** - upload section
2. **Results** - po analizie z twarzą + wykrytą emocją
3. **Sidebar** - About section

Zapisz jako `demo_screenshot.png` i wrzuć do repo.

---

### **3. LinkedIn Post**
```
🎉 Excited to share my latest project: Emotion Detection AI! 😊

Built a real-time facial emotion recognition system using:
- Custom CNN trained on 35k images
- PyTorch for deep learning
- OpenCV for face detection
- Streamlit for web deployment

Key achievements:
✅ 59.64% accuracy across 7 emotion classes
✅ ~11M parameter CNN architecture
✅ Full ML pipeline: data → training → production
✅ Deployed on Streamlit Cloud

Try it yourself: [LINK]
GitHub: [LINK]

This project demonstrates end-to-end ML development from scratch, including model architecture design, training optimization, and cloud deployment.

#MachineLearning #DeepLearning #ComputerVision #AI #Python #PyTorch #DataScience

[Dodaj screenshots]
