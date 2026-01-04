import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pandas as pd

#streamlit
st.set_page_config(
    page_title="Image Classification with ResNet18",
    layout="centered"
)

st.title("Image Classification Web App")
st.write("This application uses a pretrained ResNet18 model to classify images.")

#load model
device = torch.device("cpu")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()
model.to(device)

#image preprocessing
weights = models.ResNet18_Weights.DEFAULT
preprocess = weights.transforms()

labels = weights.meta["categories"]

#image uploader
uploaded_file = st.file_uploader(
    "Upload an image (JPG or PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_idx = torch.topk(probabilities, 5)

    results = {
        "Class": [labels[i] for i in top5_idx],
        "Probability": [float(p) for p in top5_prob]
    }

    df = pd.DataFrame(results)

    st.subheader("Top 5 Predictions")
    st.table(df)

    #visualization
    st.subheader("Prediction Probability Distribution")
    st.bar_chart(df.set_index("Class"))
