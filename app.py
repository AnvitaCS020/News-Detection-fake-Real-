
import streamlit as st
import pickle, re, os, time
import nltk
from nltk.corpus import stopwords
from nltk.stem   import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "TruthLens — Fake News Detector",
    page_icon  = "🔬",
    layout     = "centered",
    initial_sidebar_state = "collapsed"
)

# ── Full CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
  background-color: #0a0a0f !important;
  color: #e8e8f0 !important;
}

.stApp {
  background: linear-gradient(135deg, #0a0a0f 0%, #0d0d1a 50%, #0a0f1a 100%) !important;
  min-height: 100vh;
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 780px !important; margin: 0 auto; }
section[data-testid="stSidebar"] { display: none; }

/* ── Hero Header ── */
.hero {
  text-align: center;
  padding: 52px 24px 32px;
  position: relative;
}
.hero-badge {
  display: inline-flex; align-items: center; gap: 6px;
  background: rgba(99,102,241,0.15);
  border: 1px solid rgba(99,102,241,0.35);
  border-radius: 999px;
  padding: 5px 14px;
  font-size: 11px; font-weight: 600; letter-spacing: 0.08em;
  color: #a5b4fc; text-transform: uppercase;
  margin-bottom: 20px;
}
.hero-title {
  font-size: 48px; font-weight: 800; line-height: 1.1;
  background: linear-gradient(135deg, #ffffff 0%, #a5b4fc 50%, #818cf8 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text; margin-bottom: 14px;
}
.hero-sub {
  font-size: 16px; color: #6b7280; font-weight: 400; line-height: 1.6;
  max-width: 500px; margin: 0 auto 10px;
}
.hero-stats {
  display: flex; justify-content: center; gap: 32px;
  margin-top: 24px;
}
.hero-stat {
  text-align: center;
}
.hero-stat-val {
  font-size: 22px; font-weight: 700; color: #a5b4fc;
  display: block;
}
.hero-stat-lab {
  font-size: 11px; color: #4b5563; font-weight: 500;
  letter-spacing: 0.05em; text-transform: uppercase;
}

/* ── Divider ── */
.divider {
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(99,102,241,0.3), transparent);
  margin: 0 24px 32px;
}

/* ── Input Card ── */
.input-card {
  margin: 0 20px 24px;
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 20px;
  padding: 28px;
  backdrop-filter: blur(12px);
}
.input-label {
  font-size: 13px; font-weight: 600; color: #9ca3af;
  letter-spacing: 0.06em; text-transform: uppercase;
  margin-bottom: 12px; display: block;
}

/* ── Sample pill buttons ── */
.stButton > button {
  background: rgba(99,102,241,0.1) !important;
  border: 1px solid rgba(99,102,241,0.3) !important;
  color: #a5b4fc !important;
  border-radius: 999px !important;
  font-size: 12px !important; font-weight: 600 !important;
  padding: 6px 16px !important;
  transition: all 0.2s ease !important;
  width: 100% !important;
  letter-spacing: 0.03em !important;
}
.stButton > button:hover {
  background: rgba(99,102,241,0.22) !important;
  border-color: rgba(99,102,241,0.6) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 20px rgba(99,102,241,0.2) !important;
}

/* Classify (primary) button */
[data-testid="stButton"] button[kind="primary"],
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
  border: none !important;
  color: white !important;
  border-radius: 14px !important;
  font-size: 15px !important; font-weight: 700 !important;
  padding: 14px 28px !important;
  letter-spacing: 0.02em !important;
  box-shadow: 0 4px 24px rgba(99,102,241,0.35) !important;
}
[data-testid="stButton"] button[kind="primary"]:hover {
  box-shadow: 0 6px 32px rgba(99,102,241,0.55) !important;
  transform: translateY(-2px) !important;
}

/* ── Textarea ── */
.stTextArea textarea {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
  border-radius: 14px !important;
  color: #000000 !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 14px !important; line-height: 1.65 !important;
  resize: vertical !important;
  transition: border-color 0.2s !important;
}
.stTextArea textarea:focus {
  border-color: rgba(99,102,241,0.6) !important;
  box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
}
.stTextArea textarea::placeholder { color: #4b5563 !important; }
.stTextArea label { display: none !important; }

/* ── Result Boxes ── */
.result-wrapper { margin: 0 20px 24px; }

.result-fake {
  background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(220,38,38,0.18));
  border: 1.5px solid rgba(239,68,68,0.4);
  border-radius: 20px; padding: 28px 32px;
  text-align: center; position: relative; overflow: hidden;
}
.result-fake::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 3px;
  background: linear-gradient(90deg, #ef4444, #dc2626, #ef4444);
}
.result-real {
  background: linear-gradient(135deg, rgba(34,197,94,0.10), rgba(22,163,74,0.16));
  border: 1.5px solid rgba(34,197,94,0.35);
  border-radius: 20px; padding: 28px 32px;
  text-align: center; position: relative; overflow: hidden;
}
.result-real::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 3px;
  background: linear-gradient(90deg, #22c55e, #16a34a, #22c55e);
}
.result-icon { font-size: 44px; margin-bottom: 8px; line-height: 1; }
.result-label-fake {
  font-size: 28px; font-weight: 800; color: #fca5a5;
  letter-spacing: -0.02em; margin-bottom: 4px;
}
.result-label-real {
  font-size: 28px; font-weight: 800; color: #86efac;
  letter-spacing: -0.02em; margin-bottom: 4px;
}
.result-desc {
  font-size: 13px; color: #9ca3af; font-weight: 400;
}

/* ── Confidence Strip ── */
.conf-strip {
  display: grid; grid-template-columns: 1fr 1fr 1fr 1fr;
  gap: 12px; margin: 0 20px 24px;
}
.conf-card {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 14px; padding: 16px 12px; text-align: center;
}
.conf-card-val { font-size: 22px; font-weight: 700; margin-bottom: 4px; }
.conf-card-lab {
  font-size: 10px; color: #6b7280; font-weight: 600;
  letter-spacing: 0.08em; text-transform: uppercase;
}

/* ── Probability Bar ── */
.prob-section { margin: 0 20px 24px; }
.prob-header {
  font-size: 11px; font-weight: 700; color: #6b7280;
  letter-spacing: 0.08em; text-transform: uppercase;
  margin-bottom: 14px;
}
.prob-row { margin-bottom: 12px; }
.prob-row-header {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 6px;
}
.prob-label { font-size: 13px; font-weight: 600; }
.prob-value { font-size: 13px; font-weight: 700; }
.prob-track {
  height: 8px; background: rgba(255,255,255,0.06);
  border-radius: 999px; overflow: hidden;
}
.prob-fill-fake {
  height: 100%; border-radius: 999px;
  background: linear-gradient(90deg, #ef4444, #dc2626);
  transition: width 0.6s cubic-bezier(0.4,0,0.2,1);
}
.prob-fill-real {
  height: 100%; border-radius: 999px;
  background: linear-gradient(90deg, #22c55e, #16a34a);
  transition: width 0.6s cubic-bezier(0.4,0,0.2,1);
}

/* ── Info Expander ── */
.streamlit-expanderHeader {
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid rgba(255,255,255,0.07) !important;
  border-radius: 12px !important;
  color: #9ca3af !important;
  font-size: 13px !important;
}
.streamlit-expanderContent {
  background: rgba(255,255,255,0.02) !important;
  border: 1px solid rgba(255,255,255,0.05) !important;
  border-top: none !important;
  border-radius: 0 0 12px 12px !important;
}

/* ── Pipeline Steps ── */
.pipeline {
  display: flex; gap: 8px; margin: 14px 0;
  flex-wrap: wrap;
}
.pipeline-step {
  background: rgba(99,102,241,0.1);
  border: 1px solid rgba(99,102,241,0.25);
  border-radius: 8px; padding: 8px 12px;
  font-size: 12px; color: #a5b4fc; font-weight: 500;
  display: flex; align-items: center; gap: 6px;
}
.pipeline-num {
  background: rgba(99,102,241,0.3);
  border-radius: 50%; width: 18px; height: 18px;
  display: inline-flex; align-items: center; justify-content: center;
  font-size: 10px; font-weight: 700;
}

/* ── Signal Tags ── */
.signal-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }
.tag-fake {
  background: rgba(239,68,68,0.12); border: 1px solid rgba(239,68,68,0.3);
  color: #fca5a5; border-radius: 6px; padding: 3px 10px;
  font-size: 11px; font-weight: 600;
}
.tag-real {
  background: rgba(34,197,94,0.10); border: 1px solid rgba(34,197,94,0.3);
  color: #86efac; border-radius: 6px; padding: 3px 10px;
  font-size: 11px; font-weight: 600;
}

/* ── History Section ── */
.history-header {
  font-size: 11px; font-weight: 700; color: #6b7280;
  letter-spacing: 0.08em; text-transform: uppercase;
  margin: 0 20px 12px;
}
.history-item {
  margin: 0 20px 8px;
  background: rgba(255,255,255,0.02);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 12px; padding: 12px 16px;
  display: flex; align-items: center; justify-content: space-between;
  gap: 12px;
}
.history-text {
  font-size: 12px; color: #9ca3af; flex: 1;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.badge-fake {
  background: rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.35);
  color: #fca5a5; border-radius: 6px; padding: 2px 9px;
  font-size: 10px; font-weight: 700; white-space: nowrap;
}
.badge-real {
  background: rgba(34,197,94,0.12); border: 1px solid rgba(34,197,94,0.3);
  color: #86efac; border-radius: 6px; padding: 2px 9px;
  font-size: 10px; font-weight: 700; white-space: nowrap;
}

/* ── Footer ── */
.footer {
  text-align: center; padding: 32px 24px 40px;
  font-size: 12px; color: #374151;
}
.footer a { color: #6366f1; text-decoration: none; }

/* ── Spinner override ── */
.stSpinner > div { border-top-color: #6366f1 !important; }

/* ── Warning / Info ── */
.stAlert { border-radius: 12px !important; }

/* scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #1f2937; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_model.pkl')
    if not os.path.exists(model_path):
        st.error("⚠️ saved_model.pkl not found. Run `python fake_news_classifier.py` first.")
        st.stop()
    with open(model_path, 'rb') as f:
        return pickle.load(f)

payload    = load_model()
model      = payload['model']
vectorizer = payload['vectorizer']
model_name = payload['model_name']
accuracy   = payload.get('accuracy', 0.9961)

STOP_WORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text   = str(text).lower()
    text   = re.sub(r'https?://\S+|www\.\S+', '', text)
    text   = re.sub(r'\[.*?\]', '', text)
    text   = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in STOP_WORDS and len(t) > 2]
    return ' '.join(tokens)

def predict(raw_text):
    cleaned = clean_text(raw_text)
    vec     = vectorizer.transform([cleaned])
    pred    = model.predict(vec)[0]
    proba   = model.predict_proba(vec)[0]
    return int(pred), proba[0]*100, proba[1]*100

# ── Session state ─────────────────────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ''
if 'result' not in st.session_state:
    st.session_state['result'] = None

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <div class="hero-badge">🔬 AI-Powered · NLP · TF-IDF</div>
  <div class="hero-title">TruthLens</div>
  <div class="hero-sub">
    Instantly detect fake and real news articles using
    Machine Learning trained on 44,898 real-world articles.
  </div>
  <div class="hero-stats">
    <div class="hero-stat">
      <span class="hero-stat-val">{accuracy*100:.1f}%</span>
      <span class="hero-stat-lab">Accuracy</span>
    </div>
    <div class="hero-stat">
      <span class="hero-stat-val">44,898</span>
      <span class="hero-stat-lab">Articles Trained</span>
    </div>
    <div class="hero-stat">
      <span class="hero-stat-val">50K</span>
      <span class="hero-stat-lab">TF-IDF Features</span>
    </div>
    <div class="hero-stat">
      <span class="hero-stat-val">{model_name.split()[0]}</span>
      <span class="hero-stat-lab">Model</span>
    </div>
  </div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ── INPUT CARD ────────────────────────────────────────────────────────────────
st.markdown('<div class="input-card">', unsafe_allow_html=True)
st.markdown('<span class="input-label">📋 Try a sample</span>', unsafe_allow_html=True)

SAMPLE_FAKE = (
    "SHOCKING TRUTH: Government has been secretly poisoning the water supply with "
    "mind-control chemicals for decades. Whistleblowers risk their lives to expose "
    "this explosive conspiracy. Share this before it gets deleted by the deep state!"
)
SAMPLE_REAL = (
    "WASHINGTON (Reuters) - The Federal Reserve raised its benchmark interest rate "
    "by a quarter of a percentage point on Wednesday, citing continued progress on "
    "inflation. Fed Chair Jerome Powell said the committee remains data-dependent "
    "and will assess incoming economic data carefully before making further decisions."
)

c1, c2 = st.columns(2)
with c1:
    if st.button("📰 Sample Fake Article"):
        st.session_state['user_input'] = SAMPLE_FAKE
        st.session_state['result'] = None
        st.rerun()
with c2:
    if st.button("✅ Sample Real Article"):
        st.session_state['user_input'] = SAMPLE_REAL
        st.session_state['result'] = None
        st.rerun()

st.markdown('<span class="input-label" style="margin-top:20px;display:block;">✏️ Your article</span>',
            unsafe_allow_html=True)

user_input = st.text_area(
    label       = "article",
    value       = st.session_state.get('user_input', ''),
    height      = 190,
    placeholder = "Paste a news headline or full article here…",
    label_visibility = "collapsed"
)

word_count = len(user_input.split()) if user_input.strip() else 0
char_count = len(user_input)

colA, colB, colC = st.columns([3, 1, 1])
with colB:
    st.markdown(f"<p style='font-size:11px;color:#4b5563;text-align:right;margin-top:6px'>"
                f"{word_count} words</p>", unsafe_allow_html=True)
with colC:
    st.markdown(f"<p style='font-size:11px;color:#4b5563;text-align:right;margin-top:6px'>"
                f"{char_count} chars</p>", unsafe_allow_html=True)

classify_clicked = st.button("🔍 Classify Article", type="primary", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)


# ── CLASSIFY ─────────────────────────────────────────────────────────────────
if classify_clicked:
    if not user_input.strip():
        st.warning("Please enter or paste an article to classify.")
    else:
        with st.spinner("Analysing article…"):
            time.sleep(0.5)   # brief UX pause so spinner is visible
            pred, fake_pct, real_pct = predict(user_input)
        st.session_state['result']     = (pred, fake_pct, real_pct)
        st.session_state['user_input'] = user_input
        # save to history (keep last 5)
        preview = user_input[:60] + ('…' if len(user_input) > 60 else '')
        st.session_state['history'].insert(0, {
            'text': preview, 'pred': pred,
            'fake': fake_pct, 'real': real_pct
        })
        st.session_state['history'] = st.session_state['history'][:5]

# ── SHOW RESULT ───────────────────────────────────────────────────────────────
if st.session_state.get('result') is not None:
    pred, fake_pct, real_pct = st.session_state['result']
    conf = max(fake_pct, real_pct)

    # Result box
    if pred == 0:
        st.markdown(f"""
        <div class="result-wrapper">
          <div class="result-fake">
            <div class="result-icon">🚨</div>
            <div class="result-label-fake">FAKE NEWS DETECTED</div>
            <div class="result-desc">This article shows strong signals of misinformation</div>
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-wrapper">
          <div class="result-real">
            <div class="result-icon">✅</div>
            <div class="result-label-real">REAL NEWS</div>
            <div class="result-desc">This article appears to be credible and factual</div>
          </div>
        </div>""", unsafe_allow_html=True)

    # Confidence cards
    verdict_color = "#fca5a5" if pred == 0 else "#86efac"
    verdict_text  = "FAKE" if pred == 0 else "REAL"
    st.markdown(f"""
    <div class="conf-strip">
      <div class="conf-card">
        <div class="conf-card-val" style="color:#a5b4fc">{conf:.1f}%</div>
        <div class="conf-card-lab">Confidence</div>
      </div>
      <div class="conf-card">
        <div class="conf-card-val" style="color:#fca5a5">{fake_pct:.1f}%</div>
        <div class="conf-card-lab">Fake Prob.</div>
      </div>
      <div class="conf-card">
        <div class="conf-card-val" style="color:#86efac">{real_pct:.1f}%</div>
        <div class="conf-card-lab">Real Prob.</div>
      </div>
      <div class="conf-card">
        <div class="conf-card-val" style="color:{verdict_color}">{verdict_text}</div>
        <div class="conf-card-lab">Verdict</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Probability bars
    st.markdown(f"""
    <div class="prob-section">
      <div class="prob-header">Probability Breakdown</div>
      <div class="prob-row">
        <div class="prob-row-header">
          <span class="prob-label" style="color:#fca5a5">🔴 Fake News</span>
          <span class="prob-value" style="color:#fca5a5">{fake_pct:.2f}%</span>
        </div>
        <div class="prob-track">
          <div class="prob-fill-fake" style="width:{fake_pct}%"></div>
        </div>
      </div>
      <div class="prob-row">
        <div class="prob-row-header">
          <span class="prob-label" style="color:#86efac">🟢 Real News</span>
          <span class="prob-value" style="color:#86efac">{real_pct:.2f}%</span>
        </div>
        <div class="prob-track">
          <div class="prob-fill-real" style="width:{real_pct}%"></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # How it works expander
    with st.expander("🔎 How does this work?"):
        st.markdown("""
<div style="padding:4px 0">
  <p style="font-size:13px;color:#9ca3af;margin-bottom:14px;line-height:1.6">
    This classifier uses a 3-step NLP pipeline trained on 44,898 real-world news articles.
  </p>
  <div class="pipeline">
    <div class="pipeline-step"><span class="pipeline-num">1</span>Text Preprocessing</div>
    <div class="pipeline-step"><span class="pipeline-num">2</span>TF-IDF Vectorization</div>
    <div class="pipeline-step"><span class="pipeline-num">3</span>Logistic Regression</div>
  </div>
  <p style="font-size:12px;color:#6b7280;margin:14px 0 8px;font-weight:600;text-transform:uppercase;letter-spacing:0.06em">Preprocessing steps</p>
  <p style="font-size:13px;color:#9ca3af;line-height:1.7">
    Lowercase → Remove URLs → Remove punctuation → Tokenize → Remove stopwords → Lemmatize
  </p>
  <p style="font-size:12px;color:#6b7280;margin:14px 0 8px;font-weight:600;text-transform:uppercase;letter-spacing:0.06em">🔴 Fake News Signals</p>
  <div class="signal-row">
    <span class="tag-fake">shocking</span><span class="tag-fake">whistleblower</span>
    <span class="tag-fake">deep state</span><span class="tag-fake">share before deleted</span>
    <span class="tag-fake">they don't want you to know</span><span class="tag-fake">exposed</span>
    <span class="tag-fake">bombshell</span><span class="tag-fake">mainstream media</span>
  </div>
  <p style="font-size:12px;color:#6b7280;margin:14px 0 8px;font-weight:600;text-transform:uppercase;letter-spacing:0.06em">🟢 Real News Signals</p>
  <div class="signal-row">
    <span class="tag-real">reuters</span><span class="tag-real">officials confirmed</span>
    <span class="tag-real">according to</span><span class="tag-real">said in a statement</span>
    <span class="tag-real">percent</span><span class="tag-real">published</span>
    <span class="tag-real">committee</span><span class="tag-real">federal</span>
  </div>
</div>
        """, unsafe_allow_html=True)

# ── HISTORY ───────────────────────────────────────────────────────────────────
if st.session_state['history']:
    st.markdown('<div class="divider" style="margin:24px 20px 20px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="history-header">🕓 Recent Classifications</div>', unsafe_allow_html=True)
    for item in st.session_state['history']:
        badge = (f'<span class="badge-fake">FAKE</span>'
                 if item['pred'] == 0
                 else f'<span class="badge-real">REAL</span>')
        conf_val = max(item['fake'], item['real'])
        st.markdown(f"""
        <div class="history-item">
          <span class="history-text">{item['text']}</span>
          <span style="font-size:11px;color:#4b5563;white-space:nowrap">{conf_val:.0f}%</span>
          {badge}
        </div>
        """, unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  TruthLens &nbsp;·&nbsp; Built with Python, NLTK, Scikit-Learn & Streamlit<br>
  <span style="color:#1f2937">Dataset: Kaggle Fake and Real News · 44,898 articles · 99.6% accuracy</span>
</div>
""", unsafe_allow_html=True)