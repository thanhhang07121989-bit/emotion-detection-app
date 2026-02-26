
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import re
import os
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Emotion Detection CNN", page_icon="🤖", layout="wide")

st.title("🤖 Phân Tích Cảm Xúc Văn Bản")
st.markdown("**Model:** CNN | **Nhãn:** 28 cảm xúc | **Dataset:** GoEmotions")
st.markdown("---")

# Load model - Cached để chỉ load 1 lần
@st.cache_resource
def load_models():
    try:
        model = keras.models.load_model('model_cnn.h5')
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        label_map = pd.read_csv('label_map.csv')
        return model, tokenizer, label_map
    except FileNotFoundError as e:
        st.error(f"❌ Lỗi: File không tìm thấy - {str(e)}")
        st.info("📁 Cần upload: model_cnn.h5, tokenizer.pkl, label_map.csv")
        return None, None, None
    except Exception as e:
        st.error(f"❌ Lỗi: {str(e)}")
        return None, None, None

def normalize_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = " ".join(text.split())
    return text

# Load models
with st.spinner("⏳ Đang tải CNN..."):
    model, tokenizer, label_map = load_models()

if model is None:
    st.stop()

st.success("✅ Model sẵn sàng!")

# Giao diện
col1, col2 = st.columns([1.2, 1], gap="large")

with col1:
    st.subheader("📝 NHẬP VĂN BẢN")
    user_text = st.text_area(
        label="Nhập câu tiếng Anh",
        placeholder="Ví dụ: I am so happy and grateful today!",
        height=250,
        label_visibility="collapsed"
    )
    
    st.subheader("⚙️ THRESHOLD")
    threshold = st.slider("", 0.0, 1.0, 0.5, 0.05, label_visibility="collapsed")
    st.metric("Threshold hiện tại", f"{threshold:.2f}")
    
    analyze_button = st.button("🚀 PHÂN TÍCH CẢM XÚC", use_container_width=True)

with col2:
    st.subheader("😊 KẾT QUẢ")
    
    if analyze_button and user_text:
        cleaned_text = normalize_text(user_text)
        word_count = len(cleaned_text.split())
        
        if word_count < 3:
            st.warning(f"⚠️ Text quá ngắn ({word_count} từ)")
        else:
            with st.spinner("⏳ Đang phân tích..."):
                # Tokenize
                seq = tokenizer.texts_to_sequences([cleaned_text])
                padded = pad_sequences(seq, maxlen=100)
                
                # Predict
                predictions = model.predict(padded, verbose=0)[0]
                
                # Filter
                detected_idx = np.where(predictions > threshold)[0]
                if len(detected_idx) == 0:
                    detected_idx = [np.argmax(predictions)]
                
                emotions = label_map.iloc[detected_idx]['label_name'].tolist()
                
                st.success(f"✅ Phát hiện {len(emotions)} cảm xúc")
                for e in emotions[:5]:
                    st.info(f"😊 {e.capitalize()}")

st.markdown("---")

if analyze_button and user_text:
    cleaned_text = normalize_text(user_text)
    if len(cleaned_text.split()) >= 3:
        seq = tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(seq, maxlen=100)
        predictions = model.predict(padded, verbose=0)[0]
        
        st.subheader("📊 CHI TIẾT TỪNG NHÃN")
        results_df = pd.DataFrame({
            "Cảm xúc": label_map['label_name'],
            "Xác suất (%)": (predictions * 100).round(2)
        }).sort_values("Xác suất (%)", ascending=False)
        
        st.dataframe(results_df, use_container_width=True, height=400, hide_index=True)
        st.bar_chart(results_df.head(10).set_index("Cảm xúc")["Xác suất (%)"])

st.markdown("🤖 Emotion Detection - CNN Model")
