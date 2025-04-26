# streamlit_app.py

import os
import tempfile

import streamlit as st
import pandas as pd
import pickle
from transformers import pipeline

# ─── 1. Initialize the HF ASR pipeline ──────────────────────────────────────────
asr = pipeline(
    "automatic-speech-recognition",
    model="facebook/wav2vec2-base-960h",
)

# ─── 2. Initialize the HF NER pipeline for medical‑term extraction ─────────────
ner = pipeline(
    "ner",
    model="d4data/biomedical-ner-all",
    aggregation_strategy="simple"
)

# ─── 3. Load or upload the term_scores lookup ──────────────────────────────────
term_scores_path = "output.csv"
if os.path.exists(term_scores_path) and os.path.getsize(term_scores_path) > 0:
    term_scores = pd.read_csv(term_scores_path)
else:
    st.warning("⚠️ term_scores.csv not found or empty. Please upload it.")
    uploaded = st.file_uploader("Upload term_scores.csv", type=["csv"], key="term_scores")
    if uploaded:
        try:
            term_scores = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")
            st.stop()
    else:
        st.stop()
# ─── 0. Provide a sample MP3 for download ───────────────────────────────────────
sample_path = "conversation_dr_patient.mp3"
if os.path.exists(sample_path):
    with open(sample_path, "rb") as f:
        sample_bytes = f.read()
    st.download_button(
        label="📥 Download sample patient audio",
        data=sample_bytes,
        file_name="sample_audio.mp3",
        mime="audio/mpeg"
    )
else:
    st.info("No sample audio found at 'sample_audio.mp3'")


# ─── 3a. Detect your CSV’s term & score columns and compute baselines ─────────
cols = term_scores.columns.tolist()
term_col, score_col = cols[0], cols[1]
baseline_scores = term_scores.groupby(term_col)[score_col].min().to_dict()

# ─── 4. Load your trained logistic‑regression model ─────────────────────────────
clf = pickle.load(open("disposition_model.pkl", "rb"))

# ─── 5. Load feature column names from your training data ───────────────────────
feature_cols = list(pd.read_csv("GIB_part2.csv", nrows=0).columns)
feature_cols.remove("Disposition")

# ─── 6. Define helper functions ────────────────────────────────────────────────

def extract_medical_terms(txt: str) -> list[str]:
    """Run local NER to pull distinct medical entity words."""
    entities = ner(txt)
    return list({ent["word"].lower() for ent in entities})


def build_vector(terms: list[str]) -> pd.DataFrame:
    """
    Build a 1×N feature vector:
      - If a feature was extracted, use its 'present' score
      - Otherwise use that feature’s baseline ('absent') score
    """
    vect = {}
    for feat in feature_cols:
        # Get the score that means "present"
        row = term_scores[term_scores[term_col] == feat]
        present_score = float(row[score_col].iloc[0]) if not row.empty else baseline_scores.get(feat, 1)
        # Assign either the present_score or the baseline
        vect[feat] = present_score if feat.lower() in terms else baseline_scores.get(feat, present_score)
    return pd.DataFrame([vect])


# ─── 7. Streamlit UI ───────────────────────────────────────────────────────────
st.title("GI‑Bleed Triage Assistant")
st.markdown(
    """
    1. Upload a recording of the patient's symptom description.  
    2. We'll transcribe it with Hugging Face ASR, extract medical terms locally, score them,  
    3. Then predict disposition using our logistic‑regression model.
    """
)

audio = st.file_uploader(
    "Step 1: Upload patient audio (wav, mp3, m4a)",
    type=["wav", "mp3", "m4a"]
)

if audio is not None:
    # ─── 8. Transcribe with HF pipeline ─────────────────────────────────────────
    with st.spinner("Transcribing with Hugging Face ASR…"):
        suffix = os.path.splitext(audio.name)[1]
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tf.write(audio.read()); tf.flush(); tf.close()
        transcript = asr(tf.name).get("text", "")
        os.unlink(tf.name)

    st.subheader("Transcript")
    st.write(transcript)

    # ─── 9. Extract medical terms ────────────────────────────────────────────────
    terms = extract_medical_terms(transcript)
    st.write("Extracted terms:", terms)

    # ─── 10. Build and align feature vector ───────────────────────────────────────
    X_new = build_vector(terms)
    st.write("Raw feature vector:", X_new)

    # Align columns to the model’s training features and fill any missing with baseline
    expected = list(clf.feature_names_in_)
    X_aligned = X_new.reindex(columns=expected)
    X_aligned = X_aligned.fillna(value=baseline_scores)

    st.write("Aligned feature vector:", X_aligned)

    # ─── 11. Predict disposition ────────────────────────────────────────────────
    code = clf.predict(X_aligned)[0]
    label_map = {0: "Not ICU", 1: "ICU", 2: "Inpatient"}
    label = label_map.get(code, "Unknown")

    st.subheader("Predicted Disposition")
    st.write(f"**{label}**")
