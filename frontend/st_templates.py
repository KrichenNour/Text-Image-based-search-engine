import streamlit as st
import os


def load_bootstrap():
    """Loads custom CSS and Bootstrap for styling."""
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    try:
        if os.path.exists(css_path):
            with open(css_path, encoding="utf-8") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        else:
            st.caption("Note: Custom style.css not found; using default styling.")
    except Exception:
        st.caption("Note: Could not load custom CSS; using default styling.")

    st.markdown(
        '<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">',
        unsafe_allow_html=True,
    )


def st_button(icon, url, label, iconsize):
    """Generates a styled button with an icon in the sidebar."""
    if icon == "github":
        button_code = f"""
        <a href="{url}" class="btn btn-outline-info btn-lg" role="button" aria-pressed="true" style="display: flex; align-items: center; justify-content: center; gap: 0.5em;">
            <svg xmlns="http://www.w3.org/2000/svg" width="{iconsize}" height="{iconsize}" fill="currentColor" class="bi bi-github" viewBox="0 0 16 16">
                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
            </svg>
            <span>{label}</span>
        </a>"""
        st.sidebar.markdown(button_code, unsafe_allow_html=True)


def header():
    """Displays the main header of the application."""
    header_html = """
    <div style="display: flex; align-items: center; gap: 1em; margin-bottom: 1em;">
        <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" fill="currentColor" class="bi bi-search-heart" viewBox="0 0 16 16">
            <path d="M6.5 4.482c1.664-1.673 5.825 1.254 0 5.018-5.825-3.764-1.664-6.69 0-5.018Z"/>
            <path d="M13 6.5a6.471 6.471 0 0 1-1.258 3.884c.04.05.078.1.115.15l3.85 3.85a1 1 0 0 1-1.414 1.415l-3.85-3.85a1.007 1.007 0 0 1-.15-.115A6.471 6.471 0 0 1 13 6.5ZM6.5 12a5.5 5.5 0 1 0 0-11 5.5 5.5 0 0 0 0 11Z"/>
        </svg>
        <h1 style="margin: 0;">Visual Search Engine</h1>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


def footer():
    """Displays the footer with author credits."""
    footer_html = """
    <div style="text-align: center; margin-top: 2em; color: #888;">
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)


def how_to_use():
    """Displays an expander with instructions on how to use the search engine."""
    with st.sidebar.expander("How to use", expanded=False):
        st.info(
            """
        1. **Select a mode**: Choose between Image, Text, or combined search.
        2. **Provide input**:
           - For **Image** search, upload, paste, or enter an image URL.
           - For **Text** search, type your query.
        3. **Adjust settings**: Use the slider to set the number of results.
        4. **Search**: Hit the search button and view the results!
        """
        )
