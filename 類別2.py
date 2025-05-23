import streamlit as st
import pandas as pd
from joblib import load
from collections import Counter

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

def calculate_aac(sequence):
    sequence = sequence.upper()
    length = len(sequence)
    count = Counter(sequence)
    # 小寫欄位名稱：aac_a
    return {f"aac_{aa.lower()}": count.get(aa, 0) / length for aa in AMINO_ACIDS}

def calculate_dpc(sequence):
    sequence = sequence.upper()
    dpc_count = Counter()
    for i in range(len(sequence) - 1):
        dipeptide = sequence[i:i+2]
        if all(aa in AMINO_ACIDS for aa in dipeptide):
            dpc_count[dipeptide] += 1
    total = sum(dpc_count.values())
    dpc_features = {}
    for a1 in AMINO_ACIDS:
        for a2 in AMINO_ACIDS:
            key = f"dpc_{a1.lower()}{a2.lower()}"
            dpc_features[key] = dpc_count.get(a1 + a2, 0) / total if total > 0 else 0
    return dpc_features

model = load("svm_model.pkl")

st.title("SNARE Protein Predictor")
st.write("輸入蛋白質序列（單一條）：")

seq_input = st.text_area("貼上序列（A-Z）：", height=150)

if st.button("預測"):
    if not seq_input.strip():
        st.warning("請輸入序列")
    else:
        try:
            aac_features = calculate_aac(seq_input)
            dpc_features = calculate_dpc(seq_input)
            features = {**aac_features, **dpc_features}

            X_input = pd.DataFrame([features])
            prediction = model.predict(X_input)[0]
            prob = model.predict_proba(X_input)[0][prediction]

            if prediction == 1:
                st.success(f"✅ 預測結果：這是一條 SNARE 蛋白（信心值 {prob:.2f}）")
            else:
                st.info(f"❌ 預測結果：這不是 SNARE 蛋白（信心值 {prob:.2f}）")
        except Exception as e:
            st.error(f"發生錯誤：{e}")


