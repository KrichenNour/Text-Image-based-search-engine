# 🖼️ Text-Image-Based Search Engine - Quick Start

A **hybrid visual search engine** that combines **deep learning** and **full-text search** to find visually similar images from a database of 1M+ images.

---

## 📸 What It Does

- 🔍 **Text Search**: Describe an image ("red car") → Get matching images
- 📷 **Image Search**: Upload an image → Find visually similar images
- 🔗 **URL Search**: Provide image URL → Get similar results
- 🎯 **Hybrid Search**: Image + text filters → Precise results

---

## 🏗️ Project Structure

```
├── frontend/           ← Streamlit UI
│   ├── streamlit_app.py
│   ├── st_functions.py
│   └── requirements.txt
│
├── backend/            ← FastAPI server
│   ├── main.py
│   ├── search_functions.py
│   ├── modern_feature_extractor.py (ResNet50)
│   └── requirements.txt
│
├── data/               ← Image dataset
│   ├── images/         (folders 0-99)
│   └── tags/           (descriptions)
│
└── fast_index_1m_images.py  ← Bulk indexer
```

---

## 🛠️ Technologies

| Component | Tech | Purpose |
|-----------|------|---------|
| Frontend | Streamlit | Web UI |
| Backend | FastAPI | REST API |
| Database | Elasticsearch | Vector + text search |
| ML | ResNet50 + PyTorch | Feature extraction (2048-dim) |
| GPU | CUDA | GPU acceleration |

---

## 📋 How It Works

```
Image/Text Input
    ↓
Feature Extraction (ResNet50 → 2048-dim vector)
    ↓
Elasticsearch KNN Search
    ↓
Results (ranked by similarity score)
```

---

## 🚀 Installation

```bash
# 1. Clone
git clone https://github.com/dhouib-akram/Text-Image-based-search-engine.git
cd Text-Image-based-search-engine

# 2. Virtual env
python -m venv venv
venv\Scripts\activate

# 3. Dependencies
cd backend && pip install -r requirements.txt
cd ../frontend && pip install -r requirements.txt
```

---

## ▶️ Running the Code

**Open 3 terminals and run:**

### Terminal 1: Elasticsearch
```bash
cd D:\IndexationZiedNour\elasticsearch-9.1.5-windows-x86_64\elasticsearch-9.1.5\bin
$env:ES_JAVA_HOME = $null
.\elasticsearch.bat
```
✓ Runs on `http://localhost:9200`

### Terminal 2: Backend (FastAPI)
```bash
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
✓ Runs on `http://localhost:8000`

### Terminal 3: Frontend (Streamlit)
```bash
cd D:\IndexationZiedNour\Text-Image-based-search-engine\frontend
streamlit run streamlit_app.py
```
✓ Opens at `http://localhost:8501`

---

## 🔌 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/search_by_text` | POST | Text search |
| `/search_by_image/` | POST | Image search |
| `/search_by_url/` | POST | URL search |
| `/search_by_image_and_text/` | POST | Hybrid search |
| `/index_count` | GET | Total images |

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Speed/image | ~0.3s (GPU) |
| Search time | 100-300ms |
| Supported images | 1M+ |
| Index size | ~8GB |

---

## ⚙️ Configuration

**Backend** (`backend/backend_config.py`):
```python
index_name = "images_resnet"
elastic_url = "http://localhost:9200"
```

**Frontend** (`frontend/frontend_config.py`):
```python
base_url = "http://localhost:8000"
```

---

## 🎨 UI Features

- 3 search modes (Text, Image, Hybrid)
- Adjustable results (1-50)
- Live image preview
- Similarity scores
- 2-column grid layout

---

## 📈 Bulk Indexing (Optional)

```bash
# Test (10 images)
python fast_index_1m_images.py --test

# Full (1M images)
python fast_index_1m_images.py
# Time: ~3-4 hours with GPU
```

---

## 🐛 Quick Fixes

**ES not connecting?**
```bash
curl http://localhost:9200
```

**Port 8000 in use?**
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8001
```

**GPU not detected?**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 📚 Key Files

| File | Role |
|------|------|
| `streamlit_app.py` | Main UI |
| `main.py` | API server |
| `modern_feature_extractor.py` | ResNet50 extraction |
| `search_functions.py` | Search logic |
| `fast_index_1m_images.py` | Indexing |


