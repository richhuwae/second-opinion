"""Second Opinion: Streamlit demo. Start with: streamlit run app/streamlit_app.py

Loads a pre-trained model and precomputed results from app/assets/ and
results/. Nothing is trained at startup. Build the assets with
scripts/build_demo_assets.py.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
st.set_page_config(page_title="Second Opinion", page_icon="🩺", layout="wide")

import screens  # noqa: E402  (after set_page_config)

screens.sidebar()
pages = [
    st.Page(screens.screen_try, title="Try a conversation", url_path="try", default=True),
    st.Page(screens.screen_monitor, title="Monitor", url_path="monitor"),
    st.Page(screens.screen_budget, title="Review budget", url_path="budget"),
    st.Page(screens.screen_under_the_hood, title="Under the hood", url_path="under-the-hood"),
]
st.navigation(pages).run()
