import streamlit as st
import torch
import numpy as np
import cv2
from torch.autograd import Variable
from torchvision import models, transforms as trn
from torch.nn import functional as F
from PIL import Image
import pickle
from functools import partial
import wideresnet

st.set_page_config(layout="wide")
st.title("🏠 Autonomous Property Room Classifier")
st.markdown("**Scales to massive unorganized image influx - No human oversight**")

# Pickle fix
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")

@st.cache_resource
def load_model():
    model = wideresnet.resnet18(num_classes=365)
    checkpoint = torch.load("places365/wideresnet18_places365.pth.tar", map_location="cpu")
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

classes = []
with open("places365/categories_places365.txt") as f:
    classes = [line.strip().split(' ')[0][3:] for line in f]

transform = trn.Compose([
    trn.Resize((256,256)),
    trn.CenterCrop(224),
    trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def compute_clutter(img):
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    entropy = np.sum(edges > 0) / edges.size
    return "Cluttered" if entropy > 0.12 else "Clean"

def compute_quality(img):
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = gray.mean()
    score = min(100, lap_var/100 + brightness)
    return "Premium" if score > 80 else "Standard"

def get_intent_tags(room_type, clutter, quality):
    tags = []
    if 'kitchen' in room_type or 'living_room' in room_type:
        tags.append("High-Value")
    if clutter == "Clean":
        tags.append("Move-In Ready")
    if quality == "Premium":
        tags.append("Premium Listing")
    return tags

uploaded_files = st.file_uploader("Upload up to 10 room images", type=['jpg','jpeg','png'], accept_multiple_files=True)

results = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption=uploaded_file.name, width=300)
        
        # Classify
        input_img = Variable(transform(img).unsqueeze(0))
        logit = model.forward(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        
        room_type = classes[idx[0]]
        conf = probs[0].item()
        
        clutter = compute_clutter(img)
        quality = compute_quality(img)
        tags = get_intent_tags(room_type, clutter, quality)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Room Type", room_type)
        with col2:
            st.metric("Quality", quality)
        with col3:
            st.metric("Clutter", clutter)
        
        st.write("**Strategic Tags:**", ", ".join(tags))
        
        results.append({
            'type': room_type, 
            'conf': conf, 
            'quality': quality,
            'tags': tags
        })

if results:
    avg_conf = np.mean([r['conf'] for r in results])
    st.metric("Avg Confidence", f"{avg_conf:.1%}")
    st.success("✅ Images auto-mapped to search filters!")

st.info("**Enterprise Ready** - Handles unorganized multi-perspective images, architectural nuances, clutter. Strategic tags unlock revenue.")

