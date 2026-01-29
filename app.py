import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

# Page settings
st.set_page_config(
    page_title="Emotion Detection",
    page_icon="😊",
    layout="wide"
)

# Title
st.title("😊 Emotion Detection App")
st.markdown("### OpenCV + PyTorch - Facial Emotion Recognition")
st.markdown("---")

# Model CNN
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Cache model loading
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EmotionCNN().to(device)
    
    checkpoint = torch.load('emotion_model_best.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, device

# Cache cascade loading
@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load resources
with st.spinner('Loading model...'):
    model, device = load_model()
    face_cascade = load_face_cascade()

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Emotion colors
emotion_colors = {
    'Happy': '#00FF00',
    'Sad': '#0000FF',
    'Angry': '#FF0000',
    'Surprise': '#FFFF00',
    'Fear': '#800080',
    'Disgust': '#008080',
    'Neutral': '#808080'
}

def predict_emotion(face_image):
    """Predict emotion from face image"""
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    pil_img = Image.fromarray(resized)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    img_tensor = transform(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)
    
    emotion = emotion_labels[predicted.item()]
    conf = confidence.item() * 100
    
    # All probabilities
    all_probs = probabilities[0].cpu().numpy() * 100
    
    return emotion, conf, all_probs

def process_image(image):
    """Process image and detect emotions"""
    # Convert PIL -> OpenCV
    img_array = np.array(image)
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    
    results = []
    
    for i, (x, y, w, h) in enumerate(faces):
        face_roi = img[y:y+h, x:x+w]
        emotion, confidence, all_probs = predict_emotion(face_roi)
        
        # Draw on image
        color = tuple(int(emotion_colors[emotion].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)
        
        text = f"{emotion} {confidence:.1f}%"
        cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        results.append({
            'face_number': i + 1,
            'emotion': emotion,
            'confidence': confidence,
            'probabilities': all_probs,
            'bbox': (x, y, w, h)
        })
    
    # Convert back to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img_rgb, results

# Sidebar
st.sidebar.title("ℹ️ Information")
st.sidebar.markdown("""
### How to use:
1. Upload an image (JPG/PNG)
2. Click "Analyze"
3. View results!

### Model:
- **Architecture**: CNN (4 layers)
- **Dataset**: FER-2013
- **Accuracy**: 59.64%
- **Classes**: 7 emotions

### Technologies:
- PyTorch
- OpenCV
- Streamlit
""")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        if st.button("🔍 Analyze", type="primary", use_container_width=True):
            with st.spinner('Analyzing...'):
                processed_img, results = process_image(image)
                
                # Save results in session state
                st.session_state['processed_img'] = processed_img
                st.session_state['results'] = results

with col2:
    st.subheader("📊 Results")
    
    if 'results' in st.session_state and st.session_state['results']:
        st.image(st.session_state['processed_img'], caption='Analysis Result', use_container_width=True)
        
        st.markdown("### Detected Emotions:")
        
        for result in st.session_state['results']:
            emotion = result['emotion']
            conf = result['confidence']
            
            st.markdown(f"**Face {result['face_number']}:** :{'green' if conf > 70 else 'orange'}[{emotion}] ({conf:.1f}%)")
            
            # Probability chart
            probs_dict = {emotion_labels[i]: result['probabilities'][i] for i in range(7)}
            st.bar_chart(probs_dict)
            st.markdown("---")
    
    elif 'results' in st.session_state and not st.session_state['results']:
        st.warning("⚠️ No faces detected in the image!")
        st.info("Try an image with a clearly visible face.")
    else:
        st.info("👈 Upload an image and click 'Analyze'")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🚀 Project: Emotion Recognition | OpenCV + PyTorch</p>
</div>
""", unsafe_allow_html=True)    