# ui/app.py
import os
import requests
import streamlit as st

# ---------- Config ----------
API_BASE = os.getenv("LEGAL_AID_API", "http://127.0.0.1:8000")
ASK_URL = f"{API_BASE}/ask"
HEALTH_URL = f"{API_BASE}/health"

st.set_page_config(
    page_title="LegalAid (BNS) – RAG",
    page_icon="⚖️",
    layout="centered",
)

# ---------- Light styling ----------
st.markdown(
    """
    <style>
      .main { padding-top: 1rem; }
      .title { font-size: 1.4rem; font-weight: 700; margin-bottom: .25rem; }
      .subtitle { color: #777; margin-bottom: 1rem; }
      .box { background: #0e1117; border: 1px solid #262730; border-radius: 14px; padding: 14px 16px; }
      .answer { line-height: 1.55; }
      .pill { display:inline-block; padding: 2px 8px; border-radius: 999px; background: #1f2937; color:#cbd5e1; font-size: 12px; margin-right:6px; }
      .muted { color:#94a3b8; font-size: 12px; }
      .footer { color:#64748b; font-size: 12px; margin-top: 12px; }
      .disclaimer { background:#111827; border:1px dashed #334155; padding:8px 10px; border-radius:10px; color:#9aa4b2; font-size: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">⚖️ LegalAid · BNS RAG</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask a question about the Bharatiya Nyaya Sanhita. Answers are grounded in your indexed PDF.</div>', unsafe_allow_html=True)

# ---------- Health check ----------
try:
    h = requests.get(HEALTH_URL, timeout=3)
    if h.ok:
        j = h.json()
        st.markdown(
            f'<span class="pill">API: OK</span><span class="pill">Jurisdiction: {j.get("jurisdiction","—")}</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown('<span class="pill">API: Unavailable</span>', unsafe_allow_html=True)
except Exception:
    st.markdown('<span class="pill">API: Unavailable</span>', unsafe_allow_html=True)

st.write("")  # spacer

# ---------- Input row ----------
with st.form("ask_form", clear_on_submit=False):
    q = st.text_input("Your question", placeholder="e.g., Quote the BNS section that defines theft and list its key elements.")
    submitted = st.form_submit_button("Ask", use_container_width=True)

# ---------- Call backend ----------
if submitted and q.strip():
    with st.spinner("Thinking…"):
        try:
            r = requests.post(ASK_URL, json={"question": q.strip()}, timeout=60)
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach API: {e}")
            st.stop()

    if not r.ok:
        st.error(f"Server error ({r.status_code}). {r.text[:300]}")
        st.stop()

    data = r.json()
    # disclaimer (if provided by backend)
    disclaimer = data.get("disclaimer")
    if disclaimer:
        st.markdown(f'<div class="disclaimer">{disclaimer}</div>', unsafe_allow_html=True)

    # answer
    answer_md = data.get("answer", "")
    st.markdown('<div class="box answer">', unsafe_allow_html=True)
    st.markdown(answer_md, unsafe_allow_html=False)
    st.markdown('</div>', unsafe_allow_html=True)

    # citations
    cites = data.get("citations", [])
    if cites:
        with st.expander("Citations"):
            for c in cites:
                ref = c.get("ref", "")
                title = c.get("title", "")
                where = c.get("where", "")
                st.markdown(f"- **{ref}** · {title} · `{where}`")

    # raw (optional)
    if "error" in data:
        st.warning(f"Generator note: {data['error'][:300]}")

    st.markdown('<div class="footer">Tip: If a section isn’t found, try a more specific query or ensure that chapter is ingested.</div>', unsafe_allow_html=True)
