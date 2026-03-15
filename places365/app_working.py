import streamlit as st
import torch
from torch.autograd import Variable
from torchvision import models, transforms as trn
from torch.nn import functional as F
from PIL import Image
import pickle
from functools import partial

st.set_page_config(layout="wide")
st.title("🏠 Property Image AI - Working")

# Pickle fix for old models
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")

@st.cache_resource
def load_model():
    model = models.resnet18(num_classes=365)
    checkpoint = torch.load("places365/resnet18_places365.pth.tar", map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

classes = []
with open("places365/categories_places365.txt") as class_file:
    classes = [line.strip().split(' ')[0][3:] for line in class_file]

transform = trn.Compose([
    trn.Resize((256,256)),
    trn.CenterCrop(224),
    trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

uploaded_files = st.file_uploader("Upload up to 10 property images", type=['jpg','jpeg','png'], accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) > 10:
        st.error("Max 10 images")
        st.stop()
    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption=uploaded_file.name, width=400)
        
        input_img = Variable(transform(img).unsqueeze(0))
        
        logit = model.forward(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        
        st.subheader(f"**Room Type: {classes[idx[0]]}** (conf: {probs[0]:.2%})")

st.success("✅ Ready! Upload property images to see room type predictions (kitchen, living_room, etc.). Reload app if needed.")

