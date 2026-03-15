import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms, models

st.set_page_config(page_title="AI Property Inspection", layout="wide")

st.title("🏠 AI Property Inspection System")

st.markdown(
"""
Upload house images and the system will analyze:

✔ Room Type  
✔ Lighting Quality  
✔ Clutter Level  
✔ Wall Crack Detection  
✔ Paint Condition  
✔ Overall Property Score
"""
)

# ---------------- LOAD MODEL ---------------- #

@st.cache_resource
def load_model():

    model = models.resnet18(num_classes=365)

    checkpoint = torch.load("resnet18_places365.pth.tar", map_location="cpu")

    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}

    model.load_state_dict(state_dict)

    model.eval()

    return model


model = load_model()

# ---------------- LABELS ---------------- #

classes = []

with open("categories_places365.txt") as f:
    for line in f:
        classes.append(line.strip().split(' ')[0][3:])

# ---------------- IMAGE TRANSFORM ---------------- #

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- LIGHTING ---------------- #

def detect_lighting(image):

    brightness = np.array(image).mean()

    if brightness < 70:
        return "Poor"
    else:
        return "Good"

# ---------------- CLUTTER ---------------- #

def detect_clutter(image):

    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    blur = cv2.GaussianBlur(gray,(5,5),0)

    edges = cv2.Canny(blur,50,150)

    density = np.sum(edges > 0)/(edges.shape[0]*edges.shape[1])

    if density > 0.12:
        return "High", edges
    else:
        return "Low", edges

# ---------------- WALL CRACK DETECTION ---------------- #

def detect_wall_cracks(image):

    img = np.array(image)

    h,w,_ = img.shape

    wall = img[0:int(h*0.65),:]

    gray = cv2.cvtColor(wall,cv2.COLOR_RGB2GRAY)

    blur = cv2.GaussianBlur(gray,(5,5),0)

    edges = cv2.Canny(blur,100,200)

    contours,_ = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    crack_boxes=[]

    for cnt in contours:

        x,y,wc,hc = cv2.boundingRect(cnt)

        area = wc*hc

        ratio = max(wc,hc)/(min(wc,hc)+1)

        if area>800 and ratio>6:

            crack_boxes.append((x,y,wc,hc))

    return crack_boxes

# ---------------- PAINT CONDITION ---------------- #

def detect_paint(image):

    img = np.array(image)

    h,w,_ = img.shape

    wall = img[0:int(h*0.65),:]

    lab = cv2.cvtColor(wall,cv2.COLOR_RGB2LAB)

    L,A,B = cv2.split(lab)

    color_std = np.std(A)+np.std(B)

    gray = cv2.cvtColor(wall,cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray,120,220)

    edge_ratio = np.sum(edges>0)/(edges.shape[0]*edges.shape[1])

    if edge_ratio <0.02 and color_std<25:
        return "Good"

    elif edge_ratio<0.05:
        return "Average"

    else:
        return "Poor"

# ---------------- PROPERTY SCORE ---------------- #

def property_score(lighting,clutter,cracks,paint):

    score=0
    good=[]
    bad=[]

    if lighting=="Good":
        score+=25
        good.append("Lighting is good")
    else:
        bad.append("Lighting is poor")

    if clutter=="Low":
        score+=25
        good.append("Room is clean")
    else:
        bad.append("Room is cluttered")

    if cracks=="No":
        score+=25
        good.append("Walls have no cracks")
    else:
        bad.append("Wall cracks detected")

    if paint=="Good":
        score+=25
        good.append("Paint condition is good")

    elif paint=="Average":
        score+=15
        good.append("Paint slightly old")

    else:
        bad.append("Paint condition is poor")

    return score,good,bad

# ---------------- IMAGE UPLOAD ---------------- #

uploaded_files = st.file_uploader(
"Upload property images (max 10)",
type=["jpg","jpeg","png"],
accept_multiple_files=True
)

results=[]
highlight_images={}

if uploaded_files:

    if len(uploaded_files)>10:
        st.error("Upload maximum 10 images")
        st.stop()

    for uploaded_file in uploaded_files:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(image,width=400,caption=uploaded_file.name)

        img = transform(image).unsqueeze(0)

        with torch.no_grad():

            outputs = model(img)

            probs = torch.nn.functional.softmax(outputs,dim=1)

            top5 = torch.topk(probs,5)

        st.subheader("Top Scene Predictions")

        for i in range(5):

            idx = top5.indices[0][i]

            conf = top5.values[0][i].item()

            st.write(classes[idx],":",round(conf,3))

        top_class = classes[top5.indices[0][0]]

        lighting = detect_lighting(image)

        clutter,edges = detect_clutter(image)

        crack_boxes = detect_wall_cracks(image)

        paint = detect_paint(image)

        img_np=np.array(image)

        if len(crack_boxes)>3:

            cracks="Yes"

            for (x,y,wc,hc) in crack_boxes:

                cv2.rectangle(img_np,(x,y),(x+wc,y+hc),(255,0,0),2)

        else:

            cracks="No"

        score,good,bad = property_score(lighting,clutter,cracks,paint)

        st.write("Lighting:",lighting)
        st.write("Clutter:",clutter)
        st.write("Wall cracks:",cracks)
        st.write("Paint condition:",paint)

        st.progress(score/100)

        st.write("Score:",score)

        highlight_images["cracks"]=img_np
        highlight_images["clutter"]=edges

        results.append({
            "score":score,
            "good":good,
            "bad":bad
        })

# ---------------- OVERALL SCORE ---------------- #

if results:

    avg_score = round(sum(r["score"] for r in results)/len(results),2)

    st.subheader("Overall Property Score")

    st.metric("Average Score",f"{avg_score}/100")

# ---------------- CHATBOT ---------------- #

st.subheader("Ask about the property")

question = st.text_input("Ask something")

if question:

    q=question.lower()

    if "good" in q:

        for r in results:
            for g in r["good"]:
                st.write("✔",g)

    elif "bad" in q:

        for r in results:
            for b in r["bad"]:
                st.write("❌",b)

    elif "where crack" in q:

        st.image(highlight_images["cracks"])

    elif "where clutter" in q:

        st.image(highlight_images["clutter"])

    else:

        st.write("Ask about cracks, clutter, paint or house quality.")