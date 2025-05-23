import streamlit as st
from Bio import SeqIO
from collections import Counter
import pickle
import pandas as pd

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

def calc_AAC(seq):
    seq = seq.upper()
    length = len(seq)
    count = Counter(seq)
    return [count.get(aa, 0) / length for aa in AA_LIST]

def calc_DPC(seq):
    seq = seq.upper()
    dpc_count = Counter()
    for i in range(len(seq) - 1):
        dipeptide = seq[i:i+2]
        if all(aa in AA_LIST for aa in dipeptide):
            dpc_count[dipeptide] += 1
    total = sum(dpc_count.values())
    return [dpc_count.get(a1 + a2, 0) / total if total > 0 else 0
            for a1 in AA_LIST for a2 in AA_LIST]

def extract_features_from_fasta(fasta_file):
    data = []
    ids = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        aac = calc_AAC(str(record.seq))
        dpc = calc_DPC(str(record.seq))
        feature_vector = aac + dpc
        data.append(feature_vector)
        ids.append(record.id)
    return ids, data

# 網頁介面開始
st.title("🧬 SNARE 蛋白預測工具")
st.write("上傳 FASTA 序列，自動預測是否為 SNARE 蛋白。")

uploaded_file = st.file_uploader("請上傳 FASTA 檔案", type=["fasta", "fa", "txt"])

if uploaded_file is not None:
    st.success("檔案上傳成功，開始處理特徵並預測...")

    # 特徵擷取
    ids, features = extract_features_from_fasta(uploaded_file)

    # 載入模型
    with open("trained_model.pkl", "rb") as f:
        model = pickle.load(f)

    # 做預測
    predictions = model.predict(features)

    # 顯示結果
    df_result = pd.DataFrame({"ID": ids, "Prediction (1=SNARE)": predictions})
    st.dataframe(df_result)

    # 下載按鈕
    csv = df_result.to_csv(index=False).encode('utf-8')
    st.download_button("📥 下載預測結果", data=csv, file_name="snare_predictions.csv", mime='text/csv')






