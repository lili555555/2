import streamlit as st
import pandas as pd
from joblib import load

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

def calculate_aac(sequence):
    sequence = sequence.upper()
    length = len(sequence)
    return {aa: sequence.count(aa) / length for aa in AMINO_ACIDS}

# 載入模型
model = load("svm_model.pkl")

# Streamlit UI
st.title("SNARE Protein Predictor")
st.write("輸入蛋白質序列（單一條）：")

seq_input = st.text_area("貼上序列（A-Z）：", height=150)

if st.button("預測"):
    if not seq_input.strip():
        st.warning("請輸入序列")
    else:
        try:
            features = calculate_aac(seq_input)
            X_input = pd.DataFrame([features])
            prediction = model.predict(X_input)[0]
            prob = model.predict_proba(X_input)[0][prediction]

            if prediction == 1:
                st.success(f"✅ 預測結果：這是一條 SNARE 蛋白（信心值 {prob:.2f}）")
            else:
                st.info(f"❌ 預測結果：這不是 SNARE 蛋白（信心值 {prob:.2f}）")
        except Exception as e:
            st.error(f"發生錯誤：{e}")



