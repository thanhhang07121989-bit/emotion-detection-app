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

# === GỘP 2 PHẦN MODEL ===
@st.cache_resource
def prepare_model():
    if not os.path.exists('model_cnn.h5'):
        if os.path.exists('model_part1.bin') and os.path.exists('model_part2.bin'):
            with open('model_part1.bin', 'rb') as f:
                data1 = f.read()
            with open('model_part2.bin', 'rb') as f:
                data2 = f.read()
            with open('model_cnn.h5', 'wb') as f:
                f.write(data1 + data2)
    return True

prepare_model()
# === HẾT GỘP ===

st.set_page_config(page_title="Emotion Detection CNN", page_icon="🤖", layout="wide")

st.title("🤖 Phân Tích Cảm Xúc Văn Bản")
st.markdown("**Model:** CNN | **Nhãn:** 28 cảm xúc | **Dataset:** GoEmotions")
st.markdown("---")

# Load model
@st.cache_resource
def load_models():
    try:
        model = keras.models.load_model('model_cnn.h5')
        
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        
        label_map = pd.read_csv('label_map.csv')
        
        return model, tokenizer, label_map
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

# Dictionary emoji phù hợp với cảm xúc (28 nhãn)
emotion_emoji = {
    'admiration': '😍',
    'amusement': '😂',
    'anger': '😠',
    'annoyance': '😤',
    'approval': '👍',
    'caring': '🤗',
    'confusion': '😕',
    'curiosity': '🤔',
    'desire': '❤️',
    'disappointment': '😞',
    'disapproval': '👎',
    'disgust': '🤮',
    'embarrassment': '😳',
    'excitement': '🎉',
    'fear': '😨',
    'gratitude': '🙏',
    'grief': '😭',
    'hope': '🌟',
    'horror': '😱',
    'joy': '😊',
    'love': '❤️',
    'neutral': '😐',
    'nostalgia': '🌅',
    'optimism': '🌈',
    'pride': '🏆',
    'realization': '💡',
    'relief': '😌',
    'remorse': '😔',
}

# Load models
with st.spinner("⏳ Đang tải CNN..."):
    model, tokenizer, label_map = load_models()

if model is None:
    st.stop()

# ====== VALIDATION ======
n_classes = int(model.output_shape[-1])
n_labels = len(label_map)

if n_labels != n_classes:
    st.error(
        f"❌ label_map.csv KHÔNG KHỚP model!\n"
        f"- Model output: {n_classes}\n"
        f"- label_map rows: {n_labels}"
    )
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
                # FIX: Thêm padding='post', truncating='post'
                padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')
                
                # Predict
                predictions = model.predict(padded, verbose=0)[0]
                
                # Filter
                detected_idx = np.where(predictions > threshold)[0]
                if len(detected_idx) == 0:
                    detected_idx = [np.argmax(predictions)]
                
                emotions = label_map.iloc[detected_idx]['label_name'].tolist()
                scores = predictions[detected_idx]
                
                st.success(f"✅ Phát hiện {len(emotions)} cảm xúc")
                for e, score in zip(emotions[:5], scores[:5]):
                    emotion_name = str(e).lower().strip()
                    emoji = emotion_emoji.get(emotion_name, '😊')
                    st.info(f"{emoji} {str(e).capitalize()} ({score*100:.1f}%)")

st.markdown("---")

if analyze_button and user_text:
    cleaned_text = normalize_text(user_text)
    if len(cleaned_text.split()) >= 3:
        try:
            seq = tokenizer.texts_to_sequences([cleaned_text])
            # FIX: Thêm padding='post', truncating='post'
            padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')
            predictions = model.predict(padded, verbose=0)[0]
            
            st.subheader("📊 CHI TIẾT TỪNG NHÃN")
            
            # Tạo DataFrame - FIX: Bỏ min_len trick
            emotion_names = label_map['label_name'].values
            scores = (predictions * 100).round(2)
            
            # Check khớp
            if len(emotion_names) != len(scores):
                st.error(f"❌ Mismatch: {len(emotion_names)} labels ≠ {len(scores)} scores")
                st.stop()
            
            results_df = pd.DataFrame({
                "Cảm xúc": emotion_names,
                "Xác suất (%)": scores
            }).sort_values("Xác suất (%)", ascending=False)
            
            st.dataframe(results_df, use_container_width=True, height=400, hide_index=True)
            
            # Biểu đồ
            try:
                chart_data = results_df.head(10).copy()
                chart_data = chart_data.set_index("Cảm xúc")
                st.bar_chart(chart_data)
            except Exception as chart_error:
                st.warning(f"⚠️ Không thể hiển thị biểu đồ: {str(chart_error)}")
                
        except Exception as e:
            st.error(f"❌ Lỗi phân tích: {str(e)}")

st.markdown("🤖 Emotion Detection - CNN Model")
