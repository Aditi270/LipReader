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

# Setup the sidebar
with st.sidebar: 
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipReader')
    st.info('This application is originally developed from the LipNet deep learning model.')

    st.markdown("### Open the app link")
    st.markdown("- **Local link (this PC)**: [http://127.0.0.1:8501](http://127.0.0.1:8501)")

st.title('LipReader') 
st.markdown("**Link:** [http://127.0.0.1:8501](http://127.0.0.1:8501)")

ensure_assets()

# Generating a list of options or videos 
options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Choose video', options)

# Generate two columns 
col1, col2 = st.columns(2)

if options: 

    # Rendering the video 
    with col1: 
        st.info('Input video (preview)')
        file_path = os.path.join('..','data','s1', selected_video)
        st.caption("On some deployments, ffmpeg is unavailable, so we show the model-view GIF instead.")


    with col2: 
        st.info('This is all the machine learning model sees when making a prediction')
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        # `video` is normalized float32 with shape (T, H, W, 1); convert for GIF writer.
        vmin = tf.reduce_min(video)
        vmax = tf.reduce_max(video)
        denom = tf.where(tf.equal(vmax - vmin, 0), tf.ones_like(vmax - vmin), vmax - vmin)
        video_uint8 = tf.cast(((video - vmin) / denom) * 255.0, tf.uint8)
        video_uint8 = tf.squeeze(video_uint8, axis=-1)
        imageio.mimsave('animation.gif', video_uint8.numpy(), fps=10)
        st.image('animation.gif', width=400)

        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # Convert prediction to text
        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
        
