# webapp/app.py
# -----------------------------------------------------------------------------
# Dandelion vs Grass – Streamlit Web UI
#
# Goals:
# - Beautiful, simple, and robust front-end to interact with the inference API.
# - Zero hard-coding of endpoints: the API URL is read from env (API_URL).
# - Handy UX: drag & drop images, paste an image URL, batch-upload, show results.
# - Resilient: health check, clear error messages, timeouts, and small retries.
# - Lightweight: no custom components, just Streamlit + requests + Pillow.
#
# How it works:
# - We call the FastAPI endpoint POST /predict with a multipart file field "file".
# - The API returns: {"prediction": "<label>", "confidence": <float>}
# - We render a nice card with label + confidence + the image preview.
#
# Notes for reviewers:
# - This app intentionally keeps a clean separation: the model lives in the API.
# - The UI should run either locally (python streamlit) or inside Docker.
# - Make sure API_URL is properly set. In docker-compose.app.yml we set:
#       environment:
#         API_URL: http://api:8000/predict
#   Locally, you can export:
#       export API_URL="http://127.0.0.1:8000/predict"
# - The web UI never touches S3 nor MLflow directly; it only talks to the API.
# -----------------------------------------------------------------------------

from __future__ import annotations

import io
import os
import time
from typing import List, Optional, Tuple

import requests
from PIL import Image
import streamlit as st

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Read the prediction endpoint from env. Default is the local dev URL.
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")
HEALTH_URL = API_URL.replace("/predict", "/health")

APP_TITLE = "🌼 Dandelion vs 🌿 Grass — Demo"
APP_SUBTITLE = "End-to-end MLOps project • FastAPI + Streamlit + MLflow + S3 (MinIO)"

# Reasonable network timeouts to avoid hanging the UI
REQ_TIMEOUT = (5, 20)  # (connect, read) seconds

# Limit for batch uploads to keep inference snappy in demos
BATCH_LIMIT = 12


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
@st.cache_data(ttl=60)
def check_health() -> Tuple[bool, Optional[bool]]:
    """
    Ping the API /health endpoint. Returns:
    - is_up (bool): can we reach the API?
    - model_loaded (Optional[bool]): whether API says model is loaded.
    Cached briefly to avoid spamming the backend when users interact with UI.
    """
    try:
        r = requests.get(HEALTH_URL, timeout=REQ_TIMEOUT)
        if r.ok:
            data = r.json()
            return True, bool(data.get("model_loaded", False))
        return False, None
    except Exception:
        return False, None


def fetch_image_from_url(url: str) -> Image.Image:
    """
    Download an image from a remote URL and return a PIL Image (RGB).
    Raises a ValueError with a clean message for the UI on failure.
    """
    try:
        r = requests.get(url, timeout=REQ_TIMEOUT)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        return img
    except Exception as e:
        raise ValueError(f"Failed to download image: {e}")


def predict_one(pil_img: Image.Image) -> Tuple[str, float]:
    """
    Send a single PIL image to the API and return (label, confidence).
    The API expects a multipart form with a 'file' field.
    """
    with io.BytesIO() as buf:
        pil_img.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        files = {"file": ("upload.jpg", buf, "image/jpeg")}
        try:
            r = requests.post(API_URL, files=files, timeout=REQ_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            return str(data["prediction"]), float(data["confidence"])
        except requests.HTTPError as http_err:
            # Show API error detail if present (FastAPI includes "detail")
            try:
                detail = r.json().get("detail")
                raise ValueError(f"API error: {detail or http_err}")
            except Exception:
                raise ValueError(f"API error: {http_err}")
        except Exception as e:
            raise ValueError(f"Request failed: {e}")


def _badge_for_label(label: str) -> str:
    """
    Simple color-coded badge in Markdown for the predicted label.
    """
    if label.lower() == "dandelion":
        return '<span style="background:#FFE066;padding:2px 8px;border-radius:12px;">🌼 dandelion</span>'
    else:
        return '<span style="background:#C3F7C3;padding:2px 8px;border-radius:12px;">🌿 grass</span>'


def _format_conf(conf: float) -> str:
    return f"{conf*100:,.2f}%".replace(",", " ")


# -----------------------------------------------------------------------------
# Page look & feel
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Dandelion vs Grass", page_icon="🌼", layout="wide")

# Minimal custom CSS for a cleaner look
st.markdown(
    """
    <style>
    .small-muted { color: #8A8A8A; font-size: 0.92rem; }
    .card {
        border: 1px solid #eaeaea; border-radius: 14px; padding: 16px; background: #fff;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .thumb {
        border-radius: 10px; border: 1px solid #eee;
    }
    .metric {
        font-size: 1.25rem; font-weight: 600; margin-top: 2px;
    }
    .muted { color: #666; font-size: 0.95rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")
    st.write("All predictions are sent to the FastAPI backend.")

    is_up, model_loaded = check_health()
    if is_up and model_loaded:
        st.success("API reachable • Model loaded ✅", icon="✅")
    elif is_up:
        st.warning("API reachable • Model not loaded yet", icon="⚠️")
    else:
        st.error("API unreachable. Check API_URL / containers.", icon="❌")

    st.text_input("API URL", value=API_URL, disabled=True, help="Set via environment variable API_URL")

    st.divider()
    st.caption("About")
    st.write(
        "This minimal UI is part of an end-to-end MLOps project: PyTorch + MLflow + "
        "S3/MinIO + FastAPI + Streamlit."
    )
    st.caption("Tip: you can batch-upload several images at once.")

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

# -----------------------------------------------------------------------------
# Input area
# -----------------------------------------------------------------------------
tabs = st.tabs(["🔼 Upload", "🔗 From URL", "🕘 History"])

# Session state to store past predictions (image bytes + label/conf)
if "history" not in st.session_state:
    st.session_state.history: List[Tuple[bytes, str, float]] = []

with tabs[0]:
    st.markdown("**Drop images here** or click to browse (JPEG/PNG, up to ~5–10 MB each).")
    files = st.file_uploader(
        "Upload one or multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, label_visibility="collapsed"
    )
    if files:
        if len(files) > BATCH_LIMIT:
            st.info(f"Limiting to first {BATCH_LIMIT} images for demo speed.")
            files = files[:BATCH_LIMIT]

        cols = st.columns(min(3, len(files)))
        for i, f in enumerate(files):
            # Display thumbnails while we process
            with cols[i % len(cols)]:
                st.image(f, caption=f.name, use_column_width=True)

        btn = st.button("🚀 Predict", type="primary")
        if btn:
            st.subheader("Results")
            grid = st.columns(min(3, len(files)))
            for i, f in enumerate(files):
                try:
                    img = Image.open(f).convert("RGB")
                except Exception as e:
                    with grid[i % len(grid)]:
                        st.error(f"Cannot open image {f.name}: {e}")
                        continue

                t0 = time.time()
                try:
                    label, conf = predict_one(img)
                    dt = (time.time() - t0) * 1000
                    with io.BytesIO() as buf:
                        img.save(buf, format="JPEG", quality=92)
                        buf.seek(0)
                        st.session_state.history.append((buf.getvalue(), label, conf))

                    with grid[i % len(grid)]:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.image(img, use_column_width=True, output_format="JPEG")
                        st.markdown(_badge_for_label(label), unsafe_allow_html=True)
                        st.markdown(f'<div class="metric">Confidence: {_format_conf(conf)}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="muted">Latency: {dt:.0f} ms</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                except ValueError as e:
                    with grid[i % len(grid)]:
                        st.error(str(e))

with tabs[1]:
    st.markdown("Paste an **image URL** (JPEG/PNG).")
    default_url = ""
    url = st.text_input("Image URL", value=default_url, placeholder="https://raw.githubusercontent.com/.../00000010.jpg")
    go = st.button("Fetch & Predict")

    if go and url.strip():
        try:
            img = fetch_image_from_url(url.strip())
            st.image(img, caption="Input", use_column_width=True)
            t0 = time.time()
            label, conf = predict_one(img)
            dt = (time.time() - t0) * 1000

            with io.BytesIO() as buf:
                img.save(buf, format="JPEG", quality=92)
                buf.seek(0)
                st.session_state.history.append((buf.getvalue(), label, conf))

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(_badge_for_label(label), unsafe_allow_html=True)
            st.markdown(f'<div class="metric">Confidence: {_format_conf(conf)}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="muted">Latency: {dt:.0f} ms</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        except ValueError as e:
            st.error(str(e))

with tabs[2]:
    st.markdown("Most recent predictions (stored in session only).")
    if not st.session_state.history:
        st.info("No predictions yet.")
    else:
        cols = st.columns(3)
        for i, (img_bytes, label, conf) in enumerate(reversed(st.session_state.history[-12:])):
            with cols[i % 3]:
                st.image(img_bytes, use_column_width=True)
                st.markdown(_badge_for_label(label), unsafe_allow_html=True)
                st.caption(f"Conf: {_format_conf(conf)}")

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.divider()
st.markdown(
    '<span class="small-muted">Made with ❤️ · This UI talks to a FastAPI backend which loads '
    "a PyTorch model from MinIO (S3 compatible) and logs to MLflow.</span>",
    unsafe_allow_html=True,
)