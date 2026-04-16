import os

# Ensure legacy Keras is used before importing TensorFlow/Keras.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

# Import all of the dependencies
import streamlit as st
import imageio 

import tensorflow as tf 
import gdown
from utils import load_data, num_to_char
from modelutil import load_model

DATA_URL = os.environ.get(
    "LIPREADER_DATA_URL",
    "https://drive.google.com/uc?id=1YlvpDLix3S-U8fd-gqRwPcWXAXm8JwjL",
)
MODEL_URL = os.environ.get("LIPREADER_MODEL_URL")  # required for deploy

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")
MODELS_DIR = os.path.join(REPO_ROOT, "models")


def _download_and_extract(url: str, dest_dir: str, out_zip: str) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    gdown.download(url, out_zip, quiet=False)
    gdown.extractall(out_zip, dest_dir)


def ensure_assets() -> None:
    # Data (videos + alignments)
    expected_data = os.path.join(DATA_DIR, "s1")
    if not os.path.isdir(expected_data):
        with st.status("Downloading dataset (first run only)...", expanded=False):
            _download_and_extract(DATA_URL, REPO_ROOT, os.path.join(REPO_ROOT, "data.zip"))

    # Model checkpoint
    expected_ckpt = os.path.join(MODELS_DIR, "checkpoint.index")
    if not os.path.isfile(expected_ckpt):
        if not MODEL_URL:
            st.error(
                "Model weights not found. For deployment, set the environment variable "
                "`LIPREADER_MODEL_URL` to a public zip containing the `checkpoint*` files."
            )
            st.stop()
        with st.status("Downloading model weights (first run only)...", expanded=False):
            _download_and_extract(MODEL_URL, REPO_ROOT, os.path.join(REPO_ROOT, "models.zip"))


# Set the layout to the streamlit app as wide 
st.set_page_config(page_title="LipReader", layout="wide")

# Global styling (subtle, modern defaults)
st.markdown(
    """
<style>
  .block-container { padding-top: 1.25rem; padding-bottom: 2.5rem; }
  [data-testid="stSidebar"] { padding-top: 1.25rem; }
  .lipreader-hero h1 { margin-bottom: 0.25rem; }
  .lipreader-muted { color: rgba(255,255,255,0.70); }
  @media (prefers-color-scheme: light) {
    .lipreader-muted { color: rgba(0,0,0,0.60); }
  }
</style>
""",
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.markdown("## LipReader")
    st.caption("A lip-reading demo based on LipNet-style architectures.")

    with st.expander("About", expanded=False):
        st.write(
            "Select a sample video and LipReader will show the model input (cropped mouth ROI) "
            "and decode the predicted text."
        )

    show_debug = st.toggle("Show debug output", value=False)


# Hero header
st.markdown(
    """
<div class="lipreader-hero">
  <h1>LipReader</h1>
  <div class="lipreader-muted">Upload-free demo using a prepackaged dataset and pretrained weights.</div>
</div>
""",
    unsafe_allow_html=True,
)
st.divider()

ensure_assets()

# Generating a list of options or videos 
options = sorted(os.listdir(os.path.join("..", "data", "s1")))
selected_video = st.selectbox(
    "Choose a sample video",
    options,
    index=0 if options else None,
    help="These are sample clips from the bundled dataset.",
)

top_row = st.columns([2, 1])
with top_row[0]:
    st.caption(f"Selected: `{selected_video}`" if selected_video else "No samples found.")
with top_row[1]:
    run_inference = st.button("Run inference", type="primary", use_container_width=True)

if options: 

    file_path = os.path.join("..", "data", "s1", selected_video)

    tabs = st.tabs(["Model view", "Prediction"])

    with tabs[0]:
        st.subheader("What the model sees")
        st.caption("Cropped mouth region over time (GIF).")

        video, annotations = load_data(tf.convert_to_tensor(file_path))

        # `video` is normalized float32 with shape (T, H, W, 1); convert for GIF writer.
        vmin = tf.reduce_min(video)
        vmax = tf.reduce_max(video)
        denom = tf.where(tf.equal(vmax - vmin, 0), tf.ones_like(vmax - vmin), vmax - vmin)
        video_uint8 = tf.cast(((video - vmin) / denom) * 255.0, tf.uint8)
        video_uint8 = tf.squeeze(video_uint8, axis=-1)

        imageio.mimsave("animation.gif", video_uint8.numpy(), fps=10)
        st.image("animation.gif", use_container_width=False, width=420)

        if show_debug:
            st.write(
                {
                    "video_shape": tuple(video.shape),
                    "video_dtype": str(video.dtype),
                    "alignment_len": int(tf.shape(annotations)[0].numpy()),
                }
            )

    with tabs[1]:
        st.subheader("Prediction")
        st.caption("Click **Run inference** to decode the selected sample.")

        if run_inference:
            with st.spinner("Running the model..."):
                model = load_model()
                yhat = model.predict(tf.expand_dims(video, axis=0), verbose=0)
                decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()

                converted_prediction = (
                    tf.strings.reduce_join(num_to_char(decoder)).numpy().decode("utf-8").strip()
                )

            st.success("Done")
            st.markdown("#### Decoded text")
            st.code(converted_prediction if converted_prediction else "(empty)", language="text")

            if show_debug:
                st.markdown("#### Raw tokens (debug)")
                st.code(str(decoder), language="text")
        else:
            st.info("Ready when you are.")
        
