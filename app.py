import streamlit as st
import numpy as np
import pandas as pd
import pickle
import re
import os
from datetime import datetime
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings

warnings.filterwarnings("ignore")

# =========================
# Helpers: CSS + UI
# =========================
def inject_css():
    st.markdown(
        """
        <style>
          /* Page background */
          .stApp {
            background: linear-gradient(180deg, #f6f9ff 0%, #eef3ff 100%);
          }

          /* Hide Streamlit default header/footer */
          header, footer { visibility: hidden; }

          /* Top navbar */
          .topbar {
            position: sticky;
            top: 0;
            z-index: 999;
            padding: 14px 18px;
            margin: -1rem -1rem 1rem -1rem;
            background: linear-gradient(90deg, #2b6cb0 0%, #3182ce 55%, #2c5282 100%);
            border-bottom: 1px solid rgba(255,255,255,.25);
            color: white;
          }
          .topbar .row {
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap: 14px;
          }
          .brand {
            display:flex;
            align-items:center;
            gap: 10px;
            font-weight: 800;
            font-size: 22px;
            letter-spacing: .2px;
          }
          .nav {
            display:flex;
            gap: 18px;
            font-weight: 600;
            opacity: .95;
          }
          .nav a {
            color: white !important;
            text-decoration: none !important;
            padding: 6px 10px;
            border-radius: 10px;
          }
          .nav a.active, .nav a:hover {
            background: rgba(255,255,255,.16);
          }
          .userchip {
            display:flex;
            align-items:center;
            gap: 10px;
            background: rgba(255,255,255,.14);
            padding: 8px 12px;
            border-radius: 999px;
            font-weight: 600;
          }
          .avatar {
            width: 28px; height: 28px;
            border-radius: 50%;
            background: rgba(255,255,255,.75);
            display:inline-block;
          }

          /* Cards */
          .card {
            background: white;
            border: 1px solid rgba(15, 23, 42, .08);
            border-radius: 16px;
            padding: 16px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, .06);
          }
          .card h3 {
            margin: 0 0 10px 0;
            font-size: 18px;
          }
          .muted {
            color: rgba(15, 23, 42, .65);
            font-size: 13px;
          }

          /* Result header */
          .result-title {
            display:flex;
            align-items:center;
            gap: 12px;
            margin-bottom: 8px;
          }
          .emoji-badge {
            width: 52px; height: 52px;
            border-radius: 16px;
            background: #ebf8ff;
            display:flex; align-items:center; justify-content:center;
            font-size: 28px;
            border: 1px solid rgba(49,130,206,.25);
          }
          .big-label {
            font-size: 30px;
            font-weight: 900;
            margin: 0;
            line-height: 1.1;
          }

          /* Horizontal bar list */
          .barrow {
            display:flex;
            align-items:center;
            gap: 10px;
            margin: 10px 0;
          }
          .barlabel {
            width: 120px;
            font-weight: 700;
          }
          .barwrap {
            flex: 1;
            background: #edf2f7;
            border-radius: 999px;
            height: 12px;
            overflow: hidden;
            border: 1px solid rgba(15,23,42,.06);
          }
          .barfill {
            height: 100%;
            border-radius: 999px;
          }
          .barpct {
            width: 60px;
            text-align:right;
            font-weight: 700;
            color: rgba(15, 23, 42, .75);
          }

          /* Small KPI */
          .kpi {
            display:flex;
            align-items:center;
            justify-content:space-between;
            padding: 12px 14px;
            border-radius: 14px;
            background: #f8fafc;
            border: 1px solid rgba(15,23,42,.06);
          }
          .kpi .v { font-weight: 900; font-size: 18px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_topbar(active="Dashboard"):
    st.markdown(
        f"""
        <div class="topbar">
          <div class="row">
            <div class="brand">
              <span style="filter: drop-shadow(0 6px 16px rgba(0,0,0,.25));">🌀</span>
              <span>Emotion Detection System</span>
            </div>
            <div class="nav">
              <a class="{ 'active' if active=='Dashboard' else '' }" href="#">Dashboard</a>
              <a class="{ 'active' if active=='Upload' else '' }" href="#">Upload</a>
              <a class="{ 'active' if active=='Reports' else '' }" href="#">Reports</a>
            </div>
            <div class="userchip">
              <span class="avatar"></span>
              <span>Welcome, User</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================
# Model prep (merge parts)
# =========================
@st.cache_resource
def prepare_model():
    if not os.path.exists("model_cnn.h5"):
        if os.path.exists("model_part1.bin") and os.path.exists("model_part2.bin"):
            with open("model_part1.bin", "rb") as f:
                data1 = f.read()
            with open("model_part2.bin", "rb") as f:
                data2 = f.read()
            with open("model_cnn.h5", "wb") as f:
                f.write(data1 + data2)
    return True


@st.cache_resource
def load_models():
    model = keras.models.load_model("model_cnn.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    label_map = pd.read_csv("label_map.csv")
    return model, tokenizer, label_map


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = " ".join(text.split())
    return text


# Emoji map (28 labels)
emotion_emoji = {
    "admiration": "😍",
    "amusement": "😂",
    "anger": "😠",
    "annoyance": "😤",
    "approval": "👍",
    "caring": "🤗",
    "confusion": "😕",
    "curiosity": "🤔",
    "desire": "❤️",
    "disappointment": "😞",
    "disapproval": "👎",
    "disgust": "🤮",
    "embarrassment": "😳",
    "excitement": "🎉",
    "fear": "😨",
    "gratitude": "🙏",
    "grief": "😭",
    "joy": "😊",
    "love": "❤️",
    "nervousness": "😰",
    "optimism": "🌈",
    "pride": "🏆",
    "realization": "💡",
    "relief": "😌",
    "remorse": "😔",
    "sadness": "😢",
    "surprise": "😲",
    "neutral": "😐",
}

# Color palette for bars (fallback)
emotion_color = {
    "joy": "#f6ad55",
    "gratitude": "#68d391",
    "sadness": "#63b3ed",
    "anger": "#fc8181",
    "fear": "#b794f4",
    "surprise": "#4fd1c5",
    "neutral": "#a0aec0",
}


# =========================
# Streamlit page
# =========================
st.set_page_config(page_title="Emotion Detection System", page_icon="🌀", layout="wide")
inject_css()
render_topbar(active="Dashboard")

prepare_model()

with st.spinner("⏳ Đang tải mô hình..."):
    try:
        model, tokenizer, label_map = load_models()
    except Exception as e:
        st.error(f"❌ Không load được model/tokenizer/label_map: {e}")
        st.stop()

# Validate label map vs model
n_classes = int(model.output_shape[-1])
n_labels = int(len(label_map))
if n_labels != n_classes:
    st.error(
        f"❌ label_map.csv KHÔNG KHỚP model!\n"
        f"- Model output: {n_classes}\n"
        f"- label_map rows: {n_labels}"
    )
    st.stop()

id2label = dict(zip(label_map["label_id"], label_map["label_name"]))
NEUTRAL_ID = 27

# Max_len: ưu tiên theo model input_shape, fallback 250
try:
    MAX_LEN = int(model.input_shape[1])
except Exception:
    MAX_LEN = 250

# Session state for "Recent Analyses"
if "recent" not in st.session_state:
    st.session_state.recent = []  # list of dicts

# Layout: left controls + right result, bottom analytics
left, right = st.columns([0.95, 1.35], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Analyze Text for Emotion")
    user_text = st.text_area("Enter your text here...", height=140, label_visibility="collapsed")

    c1, c2 = st.columns(2)
    with c1:
        threshold = st.slider("Threshold", 0.0, 1.0, 0.50, 0.05)
    with c2:
        uncertain_cutoff = st.slider("Uncertain cutoff", 0.0, 1.0, 0.45, 0.05)

    top_k = st.slider("Top-k labels", 1, 10, 3, 1)
    analyze = st.button("Analyze Text", use_container_width=True)
    st.markdown(f'<div class="muted">Model: CNN | Labels: 28 | max_len: {MAX_LEN}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Upload Audio or Image (demo UI)")
    u1, u2 = st.columns(2)
    with u1:
        st.button("🎙️ Upload Audio", use_container_width=True, disabled=True)
    with u2:
        st.button("🖼️ Upload Image", use_container_width=True, disabled=True)
    st.markdown('<div class="muted">Chức năng này chỉ là giao diện minh hoạ (model hiện tại xử lý text).</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Emotion Analysis Result")

    if analyze and user_text.strip():
        cleaned = normalize_text(user_text)
        if len(cleaned.split()) < 3:
            st.warning("⚠️ Văn bản quá ngắn (ít hơn 3 từ).")
        else:
            with st.spinner("⏳ Đang phân tích..."):
                seq = tokenizer.texts_to_sequences([cleaned])
                padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
                pred = model.predict(padded, verbose=0)[0]

                max_score = float(np.max(pred))
                max_idx = int(np.argmax(pred))

                # Filter logic
                if max_score < float(uncertain_cutoff):
                    detected_idx = [NEUTRAL_ID]
                else:
                    detected_idx = np.where(pred >= threshold)[0].tolist()
                    if len(detected_idx) == 0:
                        detected_idx = [max_idx]
                    detected_idx = sorted(detected_idx, key=lambda i: float(pred[i]), reverse=True)[: int(top_k)]

                emotions = [id2label[int(i)] for i in detected_idx]
                scores = [float(pred[int(i)]) for i in detected_idx]

                top_emotion = emotions[0]
                top_score = scores[0]
                emoji = emotion_emoji.get(str(top_emotion).lower().strip(), "😊")

                st.markdown(
                    f"""
                    <div class="result-title">
                      <div class="emoji-badge">{emoji}</div>
                      <div>
                        <div class="muted">Detected Emotion</div>
                        <div class="big-label">{str(top_emotion).capitalize()}</div>
                      </div>
                    </div>
                    <div class="muted">Confidence (top-1): {top_score*100:.1f}%</div>
                    <div style="height:10px"></div>
                    """,
                    unsafe_allow_html=True,
                )

                st.caption(
                    f"DEBUG: max_score={max_score:.3f} | max_label={id2label[max_idx]} | "
                    f"threshold={threshold:.2f} | cutoff={uncertain_cutoff:.2f} | max_len={MAX_LEN}"
                )

                # Distribution (bars) for detected labels (or top_k labels)
                st.markdown("#### Emotion Distribution")
                for e, s in zip(emotions, scores):
                    k = str(e).lower().strip()
                    bar_color = emotion_color.get(k, "#2b6cb0")
                    st.markdown(
                        f"""
                        <div class="barrow">
                          <div class="barlabel">{emotion_emoji.get(k,"😊")} {str(e).capitalize()}</div>
                          <div class="barwrap">
                            <div class="barfill" style="width:{min(max(s,0.0),1.0)*100:.1f}%; background:{bar_color};"></div>
                          </div>
                          <div class="barpct">{s*100:.1f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Save to recent analyses
                st.session_state.recent.insert(
                    0,
                    {
                        "text": user_text.strip(),
                        "detected": str(top_emotion).capitalize(),
                        "score": f"{top_score*100:.1f}%",
                        "date": datetime.now().strftime("%d/%m/%Y %H:%M"),
                    },
                )
                st.session_state.recent = st.session_state.recent[:6]

                # Full label table
                st.markdown("#### Full label probabilities (28)")
                emotion_names = label_map["label_name"].values
                prob_pct = (pred * 100).round(2)
                results_df = (
                    pd.DataFrame({"Emotion": emotion_names, "Probability (%)": prob_pct})
                    .sort_values("Probability (%)", ascending=False)
                    .reset_index(drop=True)
                )
                st.dataframe(results_df, use_container_width=True, height=320, hide_index=True)

                # Top-10 chart
                st.markdown("#### Top-10 chart")
                chart_df = results_df.head(10).set_index("Emotion")
                st.bar_chart(chart_df)

    else:
        st.info("Nhập văn bản ở bên trái và bấm **Analyze Text** để xem kết quả.")
    st.markdown("</div>", unsafe_allow_html=True)

# Bottom row: recent + KPIs
st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
b1, b2, b3 = st.columns([1.25, 0.9, 0.9], gap="large")

with b1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Recent Analyses")
    if len(st.session_state.recent) == 0:
        st.markdown('<div class="muted">Chưa có lịch sử phân tích trong session này.</div>', unsafe_allow_html=True)
    else:
        recent_df = pd.DataFrame(st.session_state.recent)[["text", "detected", "date"]]
        recent_df.columns = ["Text", "Detected Emotion", "Date"]
        st.dataframe(recent_df, use_container_width=True, height=240, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

with b2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Confidence Score")
    if analyze and user_text.strip() and "pred" in locals():
        st.markdown(
            f"""
            <div class="kpi">
              <div class="muted">Top-1 confidence</div>
              <div class="v">{top_score*100:.1f}%</div>
            </div>
            <div style="height:10px"></div>
            """,
            unsafe_allow_html=True,
        )
        # Simple pie-like approximation using Streamlit chart
        pie_df = pd.DataFrame(
            {"part": ["Top-1", "Others"], "value": [top_score, max(0.0, 1.0 - top_score)]}
        ).set_index("part")
        st.pyplot(None)  # placeholder to keep layout stable if you remove
        st.bar_chart(pie_df)
    else:
        st.markdown('<div class="muted">Chạy phân tích để xem confidence.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with b3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Emotion Insights")
    st.markdown(
        """
        <div class="muted">
          - Mood: theo nhãn top-1<br/>
          - Keywords/Advice: demo UI (nếu muốn có thật cần thêm module NLP trích keyword)
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
