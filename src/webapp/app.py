import streamlit as st
import requests
from PIL import Image
import io

# ----------------------------
# Configuration
# ----------------------------
API_URL = "http://127.0.0.1:8000/predict"  # ton API FastAPI locale

st.set_page_config(page_title="🌼 Dandelion vs Grass Classifier", layout="centered")

# ----------------------------
# Interface
# ----------------------------
st.title("🌼 Dandelion vs Grass Classifier")
st.write("Upload une image et découvre si c’est un **pissenlit** ou de l’**herbe** !")

uploaded_file = st.file_uploader("Choisis une image (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image uploadée", use_column_width=True)

    # Bouton de prédiction
    if st.button("🔮 Prédire"):
        with st.spinner("Envoi au modèle..."):
            try:
                # Conversion image -> bytes
                img_bytes = io.BytesIO()
                image.save(img_bytes, format="JPEG")
                img_bytes = img_bytes.getvalue()

                # Envoi à l'API FastAPI
                files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
                response = requests.post(API_URL, files=files)

                if response.status_code == 200:
                    result = response.json()
                    label = result["prediction"]
                    conf = result["confidence"]

                    st.success(f"**Résultat :** {label.upper()} ✅")
                    st.progress(int(conf * 100))
                    st.write(f"Confiance : {conf:.2%}")
                else:
                    st.error(f"Erreur API : {response.text}")
            except Exception as e:
                st.error(f"⚠️ Erreur : {e}")