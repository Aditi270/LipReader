## LipReader (LipNet demo) — Streamlit app

### Run locally

```bash
pip install -r requirements.txt
streamlit run LipNet/app/streamlitapp.py
```

### Deploy (Streamlit Community Cloud)

- **Main file path**: `LipNet/app/streamlitapp.py`
- **Python**: 3.12
- **Environment variables**
  - `LIPREADER_MODEL_URL`: **required**. A public URL to a zip that contains the TensorFlow checkpoint files:
    - `models/checkpoint.index`
    - `models/checkpoint.data-00000-of-00001`
    - `models/checkpoint` (optional, small text file)
  - `LIPREADER_DATA_URL`: optional. Defaults to the dataset zip used in the notebook.

On first run, the app downloads the dataset and model weights into `LipNet/data/` and `LipNet/models/`.

