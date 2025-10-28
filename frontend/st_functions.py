import requests
import base64
import frontend_config
import streamlit as st
import os
from typing import Optional
from PIL import Image


def _resolve_image_path(original_url: str, image_id: Optional[str] = None) -> Optional[str]:
    """
    Try different strategies to resolve an on-disk image path.
    
    Priority:
    1. Construct path from ImageID using the folder structure: data/images/{ImageID // 10000}/{ImageID}.jpg
    2. Try OriginalURL as absolute path
    3. Try OriginalURL relative to CWD
    4. Try OriginalURL one level up from CWD
    """
    # Strategy 1: Construct path from ImageID (most reliable for this dataset)
    if image_id:
        try:
            image_id_int = int(image_id)
            folder_num = image_id_int // 10000
            
            # Try relative to current working directory
            relative_path = os.path.join("data", "images", str(folder_num), f"{image_id}.jpg")
            if os.path.exists(relative_path):
                return relative_path
            
            # Try one level up (if running from frontend subdirectory)
            parent_path = os.path.join("..", "data", "images", str(folder_num), f"{image_id}.jpg")
            if os.path.exists(parent_path):
                return parent_path
            
            # Try absolute path from project root
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            absolute_path = os.path.join(project_root, "data", "images", str(folder_num), f"{image_id}.jpg")
            if os.path.exists(absolute_path):
                return absolute_path
        except (ValueError, TypeError):
            pass  # If ImageID is not a valid integer, fall through to URL-based strategies
    
    # Strategy 2-4: Try OriginalURL
    if not original_url:
        return None
    
    # Absolute path
    if os.path.exists(original_url):
        return original_url
    
    # Relative to CWD
    relative_path = os.path.join(os.getcwd(), original_url)
    if os.path.exists(relative_path):
        return relative_path
    
    # One level up (frontend often runs from a subfolder)
    parent_path = os.path.join(os.path.dirname(os.getcwd()), original_url)
    if os.path.exists(parent_path):
        return parent_path
    
    return None


def display_image(response, progressive: bool = True):
    """
    Render search results.
    - progressive=True will render the first hit immediately, then fill the rest iteratively to feel faster.
    """
    cols = st.columns(2)
    if response.status_code != 200:
        st.error(f"Backend request failed with status code: {response.status_code}")
        st.write("Response:", response.text)
        return

    try:
        data = response.json()
        if "resulttype" not in data:
            st.error("Invalid response format from backend")
            return

        hits = data["resulttype"]["hits"]["hits"]
        total = data["resulttype"]["hits"]["total"]["value"]

        if total == 0 or len(hits) == 0:
            st.warning("No results found")
            return

        st.success(f"Found {total} results")
        st.caption(f"Showing {len(hits)} of {total} result(s)")

        # Prepare placeholders for progressive fill-in
        placeholders = []
        for idx, _ in enumerate(hits):
            col = cols[idx % 2]
            placeholders.append(col.empty())

        def render_hit(idx: int, hit: dict):
            try:
                original_url = hit["_source"].get("OriginalURL", "")
                image_id_str = hit["_source"].get("ImageID", "?")

                # Build caption parts
                caption_parts = []
                score = hit.get("_score")
                rank_prefix = f"#{idx+1}"
                if score is not None:
                    caption_parts.append(f"{rank_prefix} • score: {score:.4f}")
                else:
                    caption_parts.append(rank_prefix)

                if image_id_str:
                    caption_parts.append(f"ID: {image_id_str}")

                tags_val = hit["_source"].get("Tags")
                if tags_val:
                    tags = tags_val if isinstance(tags_val, str) else str(tags_val)
                    if len(tags) > 50:
                        tags = tags[:50] + "..."
                    caption_parts.append(f"Tags: {tags}")

                caption = " | ".join(caption_parts)

                image_path = _resolve_image_path(original_url, image_id_str)
                if image_path:
                    image = Image.open(image_path)
                    placeholders[idx].image(image, caption=caption, use_column_width=True)
                else:
                    with placeholders[idx].container():
                        st.write(f"📷 **Image {image_id_str}**")
                        if original_url:
                            st.code(original_url, language="text")
                        if tags_val:
                            st.caption(f"Tags: {tags_val}")
                        st.warning("Image file not accessible")
            except Exception as e:
                with placeholders[idx].container():
                    st.error(f"Error displaying result #{idx+1}: {e}")

        # Render first immediately
        render_hit(0, hits[0])

        # Fill the rest progressively
        if progressive and len(hits) > 1:
            # Use st.status when available; otherwise fall back to st.spinner for older Streamlit
            if hasattr(st, "status"):
                with st.status("Loading more results…", expanded=False):
                    for i in range(1, len(hits)):
                        render_hit(i, hits[i])
                        time_sleep = 0.02 if i < 10 else 0
                        if time_sleep:
                            import time
                            time.sleep(time_sleep)
            else:
                with st.spinner("Loading more results…"):
                    for i in range(1, len(hits)):
                        render_hit(i, hits[i])
                        time_sleep = 0.02 if i < 10 else 0
                        if time_sleep:
                            import time
                            time.sleep(time_sleep)

    except Exception as e:
        st.error(f"Error processing response: {e}")
        st.write("Response data:", response.text)


def search_by_text(query, search_type, show_result):
    base_url = frontend_config.base_url
    url = base_url + "/search_by_text/"
    json = {"tags": query, "type": search_type, "number": show_result}
    response = requests.post(url, json=json)
    display_image(response)


def search_by_url(link, show_result):
    base_url = frontend_config.base_url
    url = base_url + "/search_by_url/"
    json = {"url": link, "number": show_result}
    response = requests.post(url, json=json)
    display_image(response)


def search_by_image_and_text(query, uploaded, search_type, show_result):
    base_url = frontend_config.base_url
    url = base_url + "/search_by_image_and_text/"
    image_base64 = base64.b64encode(uploaded).decode("utf-8")
    json = {
        "img": image_base64,
        "query": query,
        "type": search_type,
        "number": show_result,
    }
    response = requests.post(url, json=json)
    display_image(response)


def search_by_upload_image(uploaded, show_result):
    base_url = frontend_config.base_url
    url = base_url + "/search_by_image/"
    image_base64 = base64.b64encode(uploaded).decode("utf-8")
    json = {"img": image_base64, "number": show_result}
    response = requests.post(url, json=json)
    display_image(response)


def get_image_count():
    """Fetches the total number of indexed images from the backend."""
    try:
        base_url = frontend_config.base_url
        url = f"{base_url}/index_count"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data.get("doc_count", 0)
        else:
            return 0
    except requests.exceptions.RequestException:
        return 0
