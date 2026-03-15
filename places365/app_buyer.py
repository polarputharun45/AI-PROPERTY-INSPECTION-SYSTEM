import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image, ExifTags
from torchvision import transforms, models
import pickle
import os
import torch.nn as nn
import wideresnet

st.set_page_config(layout="wide", page_title="AI House Scout - Buyer Helper")
st.title("🏠 AI House Scout – Buy Decision Assistant")
st.markdown("""
Upload house photos → **Buy Score + Cracks/Paint Analysis**
Helps buyers spot issues FAST – "Possible to buy?"
""")

@st.cache_resource
def load_model():
    model = wideresnet.resnet18(num_classes=365)
    checkpoint = torch.load("places365/wideresnet18_places365.pth.tar", map_location=lambda storage, loc: storage, pickle_module=pickle)
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    st.success("✅ Loaded pretrained Places365 model - accurate room predictions (kitchen, bedroom, living_room)")
    model.eval()
    return model

model = load_model()

classes = []
with open("places365/categories_places365.txt") as f:
    for line in f:
        classes.append(line.strip().split(' ')[0][3:])

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def auto_rotate_image(image):
    try:
        exif = image._getexif()
        if exif:
            for tag, value in exif.items():
                decoded_tag = ExifTags.TAGS.get(tag, tag)
                if decoded_tag == 'Orientation':
                    if value == 3: return image.rotate(180)
                    elif value == 6: return image.rotate(270)
                    elif value == 8: return image.rotate(90)
    except: pass
    return image

def detect_wall_cracks(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Blur removes texture lines (like curtains)
    blur = cv2.GaussianBlur(gray, (7,7), 0)

# Edge detection - higher thresholds for real cracks
    edges = cv2.Canny(blur, 80, 150)

    # Remove straight lines (windows / frames)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=60, maxLineGap=10)
    
    mask = np.zeros_like(edges)

    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(mask,(x1,y1),(x2,y2),255,3)

    # Keep only irregular edges (cracks)
    crack_map = cv2.bitwise_and(edges, cv2.bitwise_not(mask))

    crack_pixels = np.sum(crack_map > 0)
    total_pixels = crack_map.size

crack_ratio = crack_pixels / total_pixels\n        st.write(f"Debug crack ratio: {crack_ratio*100:.2f}% ({crack_pixels} pixels)")

    annotated = cv2.cvtColor(crack_map, cv2.COLOR_GRAY2RGB)

    if crack_ratio > 0.005:\n        return "Yes (Risk)", annotated, crack_ratio\n    else:\n        return "No", annotated, crack_ratio

def detect_paint(image):
    img = np.array(image)
    h, w = img.shape[:2]
    wall = img[0:int(h*0.65), int(w*0.1):int(w*0.9)]
    gray = cv2.cvtColor(wall, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)  # Smooth noise
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(sobelx**2 + sobely**2)
    grad_var = np.var(grad)
    brightness = gray.mean()
    if grad_var < 1000:
        return "New", grad_var, brightness
    elif grad_var < 1500:
        return "Medium", grad_var, brightness
    else:
        return "Old", grad_var, brightness

def buyer_score(room_pred, cracks, crack_ratio, paint, grad_var, brightness):
    score = 70
    room_lower = room_pred.lower()
    # Room boost (more keywords)
    room_keywords = ['kitchen', 'bedroom', 'living_room', 'bathroom', 'dining_room', 'home', 'house', 'room', 'hall', 'office', 'bath']
    if any(keyword in room_lower for keyword in room_keywords):
        score += 20
    else:
        score -= 10  # Non-room penalty
    # Cracks penalty (continuous)
    score -= min(40, crack_ratio * 2000)
    # Paint condition (grad_var low=good)
    if grad_var < 200:
        score += 15  # New/smooth
    elif grad_var < 600:
        score += 5
    else:
        score -= 10
    # Brightness bonus (bright = good)
    if brightness > 130:
        score += 10
    return max(0, min(100, score))

# Upload
uploaded_files = st.file_uploader("Upload house photos (max 10)", type=["jpg","jpeg","png"], accept_multiple_files=True, help="Get buy recommendation!")

if uploaded_files:
    results = []
    for uploaded_file in uploaded_files[:10]:
        image = Image.open(uploaded_file).convert("RGB")
        image = auto_rotate_image(image)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, width=300, caption=uploaded_file.name)
        
        # Room classification
        img_t = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_t)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            top1 = torch.topk(probs, 1)
        room_pred = classes[top1.indices[0][0]]
        
        # Analysis
        cracks, crack_img, crack_ratio = detect_wall_cracks(image)
        paint_tuple = detect_paint(image)
        paint_name = paint_tuple[0]
        grad_var = paint_tuple[1]
        brightness = paint_tuple[2]
        score = buyer_score(room_pred, cracks, crack_ratio, paint_name, grad_var, brightness)
        
        with col2:
            st.metric("Predicted Room", room_pred)
            st.metric("Wall Cracks", cracks)
            st.metric("Paint Condition", paint_name)
            st.metric("Buy Score", f"{score}/100")
            color = "🟢 YES" if score > 70 else "🟡 MAYBE" if score > 50 else "🔴 NO"
            st.markdown(f"**Recommendation**: {color} {'(Fix cracks)' if 'Yes' in cracks else ''}")
        
        st.subheader("🛠️ Detected Structural Issues")
"Crack Map (White pixels = cracks) - check debug", width=400)\n        \n        results.append({"room": room_pred, "cracks": cracks, "paint": paint_name, "score": score})
    
    # Overall
    avg_score = np.mean([r["score"] for r in results])
    st.header(f"🏡 Overall Buy Recommendation: **{avg_score:.0f}/100**")
    st.metric("Average Score", f"{avg_score:.0f}/100")
    
    if avg_score > 75:
        st.success("**BUY** – Great condition, minor cosmetic if any.")
    elif avg_score > 50:
        st.warning("**NEGOTIATE** – Fix cracks/paint for discount.")
    else:
        st.error("**AVOID** – Structural issues detected.")

st.info("👆 Upload kitchen/bedroom photos to test. Cracks = red lines on walls only (ignores frames). **Buyer superpower activated!**")

if 'results' in locals() and results:
    import json
    st.download_button("💾 Download Report", json.dumps(results, indent=2), "buyer_report.json")
