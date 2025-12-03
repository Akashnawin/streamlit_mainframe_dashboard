# app.py
import streamlit as st
import pandas as pd
import os
import io
import itertools
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import time
import json

# Firebase imports
import firebase_admin
from firebase_admin import credentials, firestore

st.set_page_config(page_title="Mainframe Message Monitor", layout="wide")

st.title("Mainframe Message Monitor — Live")

# -------------------------
# Config / defaults
# -------------------------
DEFAULT_COLLECTION = "tbl_msg"

# -------------------------
# Initialize Firestore safely
# -------------------------
def init_firestore_from_secrets():
    """
    Initialize firebase_admin using a JSON string saved in Streamlit secrets
    under key FIREBASE_SERVICE_ACCOUNT. Accepts either:
      - a JSON string (the whole file contents), or
      - a parsed dict (if you put JSON directly as a dict in secrets)
    Also supports local file "serviceAccount.json" fallback for local testing.
    """
    # if already initialized, return app
    if firebase_admin._apps:
        return firebase_admin.get_app()

    # 1) Try Streamlit secrets (preferred on Streamlit Cloud)
    sa = None
    if "FIREBASE_SERVICE_ACCOUNT" in st.secrets:
        sa_raw = st.secrets["FIREBASE_SERVICE_ACCOUNT"]
        # If secrets entry is a dict already, use it; else attempt to parse JSON string
        if isinstance(sa_raw, dict):
            sa = sa_raw
        else:
            try:
                sa = json.loads(sa_raw)
            except Exception as e:
                st.error("Failed to parse FIREBASE_SERVICE_ACCOUNT JSON from Streamlit secrets.")
                st.stop()

    # 2) Fallback: local serviceAccount.json file (useful for local dev/Colab)
    if sa is None and os.path.exists("serviceAccount.json"):
        try:
            with open("serviceAccount.json", "r") as fh:
                sa = json.load(fh)
        except Exception as e:
            st.error("Failed to read local serviceAccount.json.")
            st.stop()

    if sa is None:
        st.error("Service account JSON not found in Streamlit secrets or serviceAccount.json file. See README.")
        st.stop()

    try:
        cred = credentials.Certificate(sa)
        firebase_admin.initialize_app(cred)
    except Exception as e:
        st.error(f"Failed to initialize Firebase app: {e}")
        st.stop()

    return firebase_admin.get_app()

# initialize Firestore (this will stop the app with a nice message if missing)
init_firestore_from_secrets()
db = firestore.client()

# -------------------------
# Data fetcher (cached)
# -------------------------
@st.cache_data(ttl=10)
def fetch_messages(collection=DEFAULT_COLLECTION, limit=1000):
    """
    Fetch documents from Firestore collection, newest first by TIMESTMP.
    Returns a pandas DataFrame.
    """
    try:
        docs = db.collection(collection).order_by("TIMESTMP", direction=firestore.Query.DESCENDING).limit(limit).stream()
    except Exception as e:
        # If ordering by TIMESTMP fails (missing field types), fall back to limit() without order
        try:
            docs = db.collection(collection).limit(limit).stream()
        except Exception as e2:
            st.error(f"Failed to fetch collection '{collection}': {e2}")
            return pd.DataFrame()

    rows = []
    for d in docs:
        rec = d.to_dict()
        rec["_docid"] = d.id
        rows.append(rec)

    if rows:
        df_local = pd.DataFrame(rows)
    else:
        # create typical columns so UI doesn't break
        df_local = pd.DataFrame(columns=['SEQNO','TIMESTMP','JOBNAME','MSGID','USERID','TEXT','predicted_label','pred_prob_critical','_docid'])
    return df_local

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Controls")
COLLECTION = st.sidebar.text_input("Firestore collection", value=DEFAULT_COLLECTION)
limit = st.sidebar.number_input("Documents to fetch", min_value=50, max_value=5000, value=1000, step=50)
refresh_seconds = st.sidebar.number_input("Auto-refresh (secs)", min_value=5, max_value=300, value=20)
filter_label = st.sidebar.selectbox("Filter label", ["all", "critical", "normal"])
show_wordcloud = st.sidebar.checkbox("Show wordcloud for critical", value=True)

if st.sidebar.button("Refresh now"):
    # Clear cache for fetch_messages (so new data loads immediately)
    fetch_messages.clear()
    st.experimental_rerun()

# -------------------------
# Auto-refresh timer using session_state (non-blocking)
# -------------------------
if "last_refresh" not in st.session_state:
    st.session_state["last_refresh"] = time.time()

now = time.time()
if now - st.session_state["last_refresh"] > int(refresh_seconds):
    st.session_state["last_refresh"] = now
    # Clear cache so next fetch gets fresh data
    fetch_messages.clear()
    st.experimental_rerun()

# -------------------------
# Main UI
# -------------------------
df = fetch_messages(collection=COLLECTION, limit=int(limit))
st.metric("Total messages", len(df))

col1, col2 = st.columns([3,1])

with col1:
    st.header("Latest messages")
    if df.empty:
        st.info(f"No messages in collection '{COLLECTION}'.")
    else:
        # try to present TIMESTMP nicely if present
        if 'TIMESTMP' in df.columns:
            # show most recent first
            display_df = df.sort_values(by='TIMESTMP', ascending=False).head(500)
        else:
            display_df = df.head(500)

        if filter_label != "all" and 'predicted_label' in display_df.columns:
            display_df = display_df[display_df.get('predicted_label', '') == filter_label]

        show_cols = [c for c in ['SEQNO','TIMESTMP','JOBNAME','MSGID','USERID','predicted_label','pred_prob_critical','TEXT','_docid'] if c in display_df.columns]
        st.dataframe(display_df[show_cols].fillna(""), height=600)

        # offer CSV download of currently displayed frame
        csv = display_df[show_cols].to_csv(index=False).encode('utf-8')
        st.download_button("Download displayed CSV", data=csv, file_name="tbl_msg_displayed.csv", mime="text/csv")

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

st.caption("Auto-refreshes every {}s if left open. Use 'Refresh now' to force immediate update.".format(refresh_seconds))

# -------------------------
# Helpful debug / info (collapsed)
# -------------------------
with st.expander("App info / Debug"):
    st.write("Firestore collection:", COLLECTION)
    st.write("Rows fetched:", len(df))
    if 'predicted_label' in df.columns:
        st.write("Predicted label counts:")
        st.write(df['predicted_label'].value_counts())
    st.write("Last refresh (epoch):", st.session_state.get("last_refresh"))

# End of file
