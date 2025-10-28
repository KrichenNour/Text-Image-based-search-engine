import streamlit as st
import time
import re
from st_templates import st_button, load_bootstrap, header, footer, how_to_use
import st_functions as sf
import frontend_config


def is_valid_url(url):
    """Validate if the given string is a URL."""
    url_pattern = r"https?://\S+"
    return re.match(url_pattern, url) is not None


# --- Page Setup ---
st.set_page_config(page_title="Visual Search Engine", layout="wide")
load_bootstrap()
header()

# --- Sidebar ---
with st.sidebar:
    st.markdown("## Search Controls")
    filter = st.radio("Search Mode", ("Image", "Text", "Text & Image"), horizontal=True)
    show_result = st.slider("Number of results", min_value=1, max_value=50, value=20)
    how_to_use()
    
    # Made by section - always visible at the bottom of sidebar
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; margin-top: 1em; color: #888;">
            <p>Made by<br><b>Nour Krichen & Zied Kallel</b></p>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- Main Page ---
image_count = sf.get_image_count()
st.info(f"Total images indexed: {image_count}")

# Ensure session state for query image so preview can update instantly
if "q_image_bytes" not in st.session_state:
    st.session_state.q_image_bytes = None
if "q_image_meta" not in st.session_state:
    st.session_state.q_image_meta = ""

if filter == "Text":
    with st.form("text_search_form"):
        query = st.text_input("Describe what you're looking for...", placeholder="e.g., 'a red car' or 'a sunny beach'")
        search_type = st.selectbox("Text search mode", ("fuzzy", "match"), help="Fuzzy is more lenient, match is stricter.")
        submit = st.form_submit_button("Search by Text")
        if submit and query:
            with st.spinner("Searching..."):
                start_time = time.time()
                sf.search_by_text(query, search_type, show_result)
                st.toast(f"Time taken: {time.time() - start_time:.3f}s")

elif filter == "Image":
    # Layout: left = inputs, right = live preview
    cols = st.columns([3, 1])

    # Left column inputs (uploader outside form to allow instant preview)
    with cols[0]:
        # Uploader (instant update)
        uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"], key="img_upload_image")
        if uploaded_file is not None:
            st.session_state.q_image_bytes = uploaded_file.getvalue()
            st.session_state.q_image_meta = f"{uploaded_file.name} • {uploaded_file.size/1024:.1f} KB"

        # Paste support (optional dependency) — also instant
        try:
            import importlib
            spec = importlib.util.find_spec("streamlit_paste_image")
            if spec is not None:
                paste_mod = importlib.import_module("streamlit_paste_image")
                pasted_image = paste_mod.paste_image(label="Or paste image from clipboard")
                if pasted_image is not None:
                    from io import BytesIO
                    buf = BytesIO()
                    pasted_image.save(buf, format="PNG")
                    st.session_state.q_image_bytes = buf.getvalue()
                    st.session_state.q_image_meta = f"from clipboard • {len(st.session_state.q_image_bytes)/1024:.1f} KB"
            else:
                st.caption("Tip: You can also paste an image here if supported.")
        except Exception:
            st.caption("Tip: You can also paste an image here if supported.")

        # Keep URL and Search inside a form
        with st.form("image_search_form"):
            url = st.text_input("Image URL (optional)")
            submit = st.form_submit_button("Search by Image")

    # Right column: live preview (always visible when available)
    with cols[1]:
        preview_slot = st.empty()
        meta_slot = st.empty()
        if st.session_state.q_image_bytes is not None:
            preview_slot.image(st.session_state.q_image_bytes, caption="Query image", width=320)
            if st.session_state.q_image_meta:
                meta_slot.caption(st.session_state.q_image_meta)

    # Handle search submission
    if 'submit' in locals() and submit:
        with st.spinner("Searching..."):
            if url and is_valid_url(url):
                start_time = time.time()
                sf.search_by_url(url, show_result)
                st.toast(f"Time taken: {time.time() - start_time:.3f}s")
            elif st.session_state.q_image_bytes:
                start_time = time.time()
                sf.search_by_upload_image(st.session_state.q_image_bytes, show_result)
                st.toast(f"Time taken: {time.time() - start_time:.3f}s")
            else:
                st.warning("Please provide an image via URL, upload, or paste.")

elif filter == "Text & Image":
    # Layout: left = inputs, right = live preview
    cols = st.columns([3, 1])

    # Left column
    with cols[0]:
        # Uploader outside form for instant preview
        uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"], key="img_upload_text_image")
        if uploaded_file is not None:
            st.session_state.q_image_bytes = uploaded_file.getvalue()
            st.session_state.q_image_meta = f"{uploaded_file.name} • {uploaded_file.size/1024:.1f} KB"

        # Paste support (optional)
        try:
            import importlib
            spec = importlib.util.find_spec("streamlit_paste_image")
            if spec is not None:
                paste_mod = importlib.import_module("streamlit_paste_image")
                pasted_image = paste_mod.paste_image(label="Or paste image from clipboard")
                if pasted_image is not None:
                    from io import BytesIO
                    buf = BytesIO()
                    pasted_image.save(buf, format="PNG")
                    st.session_state.q_image_bytes = buf.getvalue()
                    st.session_state.q_image_meta = f"from clipboard • {len(st.session_state.q_image_bytes)/1024:.1f} KB"
        except Exception:
            pass

        # Form for text inputs and Search button
        with st.form("combined_search_form"):
            query = st.text_input("Filter by text", placeholder="e.g., 'a person walking a dog'")
            search_type = st.selectbox("Text search mode", ("fuzzy", "multi_match"))
            url = st.text_input("Image URL (optional)")
            submit = st.form_submit_button("Search with Text and Image")

    # Right column: live preview
    with cols[1]:
        preview_slot = st.empty()
        meta_slot = st.empty()
        if st.session_state.q_image_bytes is not None:
            preview_slot.image(st.session_state.q_image_bytes, caption="Query image", width=320)
            if st.session_state.q_image_meta:
                meta_slot.caption(st.session_state.q_image_meta)

    # Handle search submission
    if 'submit' in locals() and submit:
        with st.spinner("Searching..."):
            if (url and is_valid_url(url)) and query:
                start_time = time.time()
                st.info("Combined search with URL+text is not supported; searching by URL only.")
                sf.search_by_url(url, show_result)
                st.toast(f"Time taken: {time.time() - start_time:.3f}s")
            elif st.session_state.q_image_bytes and query:
                start_time = time.time()
                sf.search_by_image_and_text(
                    uploaded=st.session_state.q_image_bytes,
                    query=query,
                    search_type=search_type,
                    show_result=show_result,
                )
                st.toast(f"Time taken: {time.time() - start_time:.3f}s")
            else:
                st.warning("Please provide both a text query and an image (URL, upload, or paste).")

footer()
