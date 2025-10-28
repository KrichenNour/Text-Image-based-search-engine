import streamlit as st
from io import BytesIO

def image_uploader_component(preview_slot=None):
    """A component for uploading, pasting, or providing an image URL."""
    q_image_bytes = None
    
    cols = st.columns([2, 2])
    with cols[0]:
        url = st.text_input("Image URL (optional)")
        uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

    with cols[1]:
        pasted_bytes = None
        try:
            import importlib
            spec = importlib.util.find_spec("streamlit_paste_image")
            if spec is not None:
                paste_mod = importlib.import_module("streamlit_paste_image")
                pasted_image = paste_mod.paste_image(label="Or paste image from clipboard", key=f"paste_{st.session_state.get('page_key', 0)}")
                if pasted_image is not None:
                    buf = BytesIO()
                    pasted_image.save(buf, format="PNG")
                    pasted_bytes = buf.getvalue()
            else:
                pass

        except Exception:
                pass

    # Determine which image source to use
    if uploaded_file:
        q_image_bytes = uploaded_file.read()
        caption = f"{uploaded_file.name} • {len(q_image_bytes)/1024:.1f} KB"
    elif pasted_bytes:
        q_image_bytes = pasted_bytes
        caption = f"Pasted image • {len(q_image_bytes)/1024:.1f} KB"
    elif url:
        # Let the main app handle URL validation and fetching
        pass
    
    if preview_slot and q_image_bytes:
        preview_slot.image(q_image_bytes, caption=caption, use_column_width=True)

    return url, q_image_bytes
