# app.py
import streamlit as st
import pandas as pd
import os
import io
import itertools
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Firebase imports
import firebase_admin
from firebase_admin import credentials, firestore

st.set_page_config(page_title="Mainframe Message Monitor", layout="wide")

st.title("Mainframe Message Monitor — Live")

# -------------------------
# Initialize Firestore safely
# -------------------------
# We will read the service account JSON from Streamlit secrets (secure)
# The secret name we'll use: FIREBASE_SERVICE_ACCOUNT (string of JSON)
def init_firestore_from_secrets():
    if firebase_admin._apps:
        return firebase_admin.get_app()
    # Read JSON string from Streamlit secrets
    if "FIREBASE_SERVICE_ACCOUNT" not in st.secrets:
        st.error("Service account JSON not found in Streamlit secrets. See README.")
        st.stop()
    sa_json = st.secrets["FIREBASE_SERVICE_ACCOUNT"]
    # write to temp in-memory file and load credentials
    cred_dict = None
    try:
        import json
        cred_dict = json.loads(sa_json)
    except Exception as e:
        st.error("Failed to parse FIREBASE_SERVICE_ACCOUNT JSON.")
        st.stop()
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)
    return firebase_admin.get_app()

# initialize
init_firestore_from_secrets()
db = firestore.client()

# -------------------------
# Data fetcher (cached)
# -------------------------
@st.cache_data(ttl=10)
def fetch_messages(limit=1000):
    docs = db.collection("tbl_msg").order_by("TIMESTMP", direction=firestore.Query.DESCENDING).limit(limit).stream()
    rows = []
    for d in docs:
        rec = d.to_dict()
        rec["_docid"] = d.id
        rows.append(rec)
    if rows:
        df_local = pd.DataFrame(rows)
    else:
        df_local = pd.DataFrame(columns=['SEQNO','TIMESTMP','JOBNAME','MSGID','USERID','TEXT','predicted_label','pred_prob_critical'])
    return df_local

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Controls")
limit = st.sidebar.number_input("Documents to fetch", min_value=50, max_value=5000, value=1000, step=50)
refresh_seconds = st.sidebar.number_input("Auto-refresh (secs)", min_value=5, max_value=300, value=20)
filter_label = st.sidebar.selectbox("Filter label", ["all", "critical", "normal"])
show_wordcloud = st.sidebar.checkbox("Show wordcloud for critical", value=True)

if st.sidebar.button("Refresh now"):
    st.experimental_rerun()

# -------------------------
# Main UI
# -------------------------
df = fetch_messages(limit=int(limit))
st.metric("Total messages", len(df))

col1, col2 = st.columns([3,1])

with col1:
    st.header("Latest messages")
    if df.empty:
        st.info("No messages in collection 'tbl_msg'.")
    else:
        display_df = df.sort_values(by='TIMESTMP', ascending=False).head(500)
        if filter_label != "all":
            display_df = display_df[display_df.get('predicted_label', '') == filter_label]
        # show columns nicely
        show_cols = [c for c in ['SEQNO','TIMESTMP','JOBNAME','MSGID','USERID','predicted_label','pred_prob_critical','TEXT'] if c in display_df.columns]
        st.dataframe(display_df[show_cols].fillna(""), height=600)

with col2:
    st.header("Analytics")
    if 'predicted_label' in df.columns:
        vc = df['predicted_label'].value_counts()
        st.subheader("Predicted label distribution")
        st.bar_chart(vc)

    # Top tokens in critical messages
    if 'predicted_label' in df.columns and not df[df['predicted_label']=='critical'].empty:
        crits = df[df['predicted_label']=='critical']['TEXT'].astype(str).str.upper().fillna('')
        tokens = crits.str.findall(r"[A-Z0-9']+")
        all_tokens = list(itertools.chain.from_iterable(tokens))
        stop = set(["THE","AND","TO","OF","IN","IS","A","AN","JOB","STEP","USER","MSG","OPS"])
        freq = Counter([t for t in all_tokens if t not in stop and len(t)>1])
        top = freq.most_common(20)
        if top:
            words, counts = zip(*top)
            st.subheader("Top tokens (critical)")
            st.bar_chart(pd.DataFrame({'token':list(words),'count':list(counts)}).set_index('token'))

        if show_wordcloud:
            text_concat = " ".join(crits.tolist())
            if text_concat.strip():
                wc = WordCloud(width=900, height=400, background_color='white').generate(text_concat)
                fig, ax = plt.subplots(figsize=(9,3))
                ax.imshow(wc, interpolation='bilinear'); ax.axis('off')
                st.pyplot(fig)

st.caption("Auto-refreshes every {}s if left open.".format(refresh_seconds))

# Auto refresh mechanism (simple)
import time
last = time.time()
if st.button("Start auto-refresh (will rerun every N secs)"):
    while True:
        time.sleep(refresh_seconds)
        st.experimental_rerun()
