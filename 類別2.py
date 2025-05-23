import streamlit as st
import pandas as pd
import numpy as np
import io
from Bio import SeqIO
from sklearn.ensemble import RandomForestClassifier
import joblib

# 載入預訓練模型（請確認這個檔案已經在 GitHub repo 裡）
@st.cache_resource
def load_model():
    return joblib.load("train_model.pkl")

# 特徵擷取（示範：AAC）
def extract_feature_vector(seq):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    counts = [seq.count(aa) for aa in amino_acids]
    total = sum(counts)
    if total == 0:
        return [0] * 20
    return [c / total for c in counts]

# 從 FASTA 檔讀入並轉成特徵矩陣
def extract_features_from_fasta(uploaded_file):
    # 將 binary 檔案轉為文字檔案
    fasta_text = io.StringIO(uploaded_file.read().decode("utf-8"))

    ids = []
    features = []

    for record in SeqIO.parse(fasta_text, "fasta"):
        ids.append(record.id)
        features.append(extract_feature_vector(str(record.seq)))

    return ids, features

# 網頁介面
st.title("SNARE Protein Predictor")
st.write("上傳 FASTA 序列，我們將幫你預測是否為 SNARE 蛋白。")

uploaded_file = st.file_uploader("選擇 FASTA 檔案", type=["fasta", "fa"])

if uploaded_file:
    ids, features = extract_features_from_fasta(uploaded_file)
    model = load_model()
    preds = model.predict(features)
    
    df = pd.DataFrame({"ID": ids, "Prediction": preds})
    df["Prediction"] = df["Prediction"].map({1: "SNARE", 0: "NOT_SNARE"})

    st.success("預測完成！")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "下載預測結果 CSV",
        csv,
        "snare_predictions.csv",
        "text/csv",
        key="download-csv"
    )
