import streamlit as st
import os
import numpy as np
import json
import uuid
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics.pairwise import cosine_similarity

# Load model
model = MobileNetV2(weights="imagenet", include_top=False, pooling='avg')

# Extract image features
def extract_features(image):
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    features = model.predict(image)
    return features.flatten()

# Save image + metadata
def save_image(img, folder, filename, meta):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    img.save(path)

    meta["image"] = path
    with open("items.json", "a") as f:
        f.write(json.dumps(meta) + "\n")

    return path

# Load embeddings
def load_embeddings(folder):
    embeddings = []
    files = []
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        if path.endswith((".jpg", ".png", ".jpeg")):
            try:
                img = Image.open(path)
                feat = extract_features(img)
                embeddings.append(feat)
                files.append(path)
            except:
                continue
    return np.array(embeddings), files


# Streamlit UI
st.set_page_config(page_title="FindItBack", layout="centered")
st.title("🎒 FindItBack – Lost & Found App (Campus Edition)")

# Search Section
st.subheader("🔎 Search Posted Items")
search_term = st.text_input("Search by name, place, or item ID")
if search_term:
    if os.path.exists("items.json"):
        st.info(f"Results for '{search_term}':")
        with open("items.json") as f:
            data = [json.loads(line) for line in f if search_term.lower() in line.lower()]
            if data:
                for item in data:
                    if os.path.exists(item["image"]):
                        st.image(item["image"], width=200)
                    else:
                        st.warning(f"Image not found: {item['image']}")
                    st.markdown(f"""
                    **Item Name:** {item.get("name", "N/A")}  
                    **Place:** {item.get("place", "N/A")}  
                    **Date:** {item.get("date", "N/A")}  
                    **Item ID:** `{item.get("id", "N/A")}`  
                    **Uploader:** {item.get("uploader_name", "N/A")}  
                    **Contact:** `{item.get("uploader_contact", "N/A")}`
                    """)
            else:
                st.warning("No matching items found.")
    else:
        st.warning("No items posted yet.")

st.markdown("---")

# Upload Section
option = st.radio("Upload a:", ["Lost Item", "Found Item"])
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
    except:
        st.error("Invalid image file!")
        st.stop()

    name = st.text_input("Item Name")
    place = st.text_input("Place where item was lost/found")
    date = st.date_input("Date Found/Lost")
    uploader_name = st.text_input("Your Name")
    uploader_contact = st.text_input("Your Contact Number")

    if st.button("Submit"):
        folder = "lost" if option == "Lost Item" else "found"
        item_id = str(uuid.uuid4())[:8]
        filename = f"{item_id}.jpg"

        meta = {
            "id": item_id,
            "type": option,
            "name": name,
            "place": place,
            "date": str(date),
            "uploader_name": uploader_name,
            "uploader_contact": uploader_contact
        }

        save_path = save_image(img, folder, filename, meta)

        st.success(f"{option} uploaded successfully!")

        # Matching
        other_folder = "found" if option == "Lost Item" else "lost"
        if len(os.listdir(other_folder)) == 0:
            st.info(f"No items in '{other_folder}' folder to match with.")
        else:
            st.subheader("🔍 Matching Results")
            query_feat = extract_features(img)
            db_feats, db_files = load_embeddings(other_folder)

            if len(db_feats) > 0:
                sims = cosine_similarity([query_feat], db_feats)[0]
                top_3 = sims.argsort()[-3:][::-1]
                for idx in top_3:
                    matched_path = db_files[idx]
                    matched_score = sims[idx]
                    st.image(matched_path, caption=f"Match Score: {matched_score:.2f}", width=300)

                    # Show match metadata
                    if os.path.exists("items.json"):
                        with open("items.json") as f:
                            for line in f:
                                item = json.loads(line)
                                if item.get("image") == matched_path:
                                    st.markdown(f"""
                                    **Item Name:** {item.get("name", "N/A")}  
                                    **Place:** {item.get("place", "N/A")}  
                                    **Date:** {item.get("date", "N/A")}  
                                    **Uploaded By:** {item.get("uploader_name", "N/A")}  
                                    **Contact:** `{item.get("uploader_contact", "N/A")}`
                                    """)
                                    break
            else:
                st.warning("No valid images to match with.")

st.markdown("---")

# Lost Items Gallery
st.subheader("📁 Lost Items Gallery")
lost_files = os.listdir("lost") if os.path.exists("lost") else []
for f in lost_files:
    if f.endswith((".jpg", ".jpeg", ".png")):
        st.image(f"lost/{f}", width=150)

# Found Items Gallery
st.subheader("📁 Found Items Gallery")
found_files = os.listdir("found") if os.path.exists("found") else []
for f in found_files:
    if f.endswith((".jpg", ".jpeg", ".png")):
        st.image(f"found/{f}", width=150)
