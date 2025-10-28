"""
Elasticsearch Setup Script for Text-Image Search Engine
This script will help you:
1. Check if Elasticsearch is running
2. Build the feature vectors with tags into a JSON file
3. Create the Elasticsearch index
4. Bulk index all the data
"""

import subprocess
import sys
import os
from pathlib import Path

def check_elasticsearch():
    """Check if Elasticsearch is running"""
    try:
        import requests
        response = requests.get("http://localhost:9200")
        if response.status_code == 200:
            print("✓ Elasticsearch is running!")
            print(f"  Version: {response.json()['version']['number']}")
            return True
        else:
            print("✗ Elasticsearch returned an error")
            return False
    except Exception as e:
        print(f"✗ Cannot connect to Elasticsearch at http://localhost:9200")
        print(f"  Error: {e}")
        return False

def install_dependencies():
    """Install required Python packages"""
    print("\n📦 Installing required packages...")
    packages = ["elasticsearch", "numpy", "requests"]
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
        print("✓ All packages installed successfully!")
        return True
    except Exception as e:
        print(f"✗ Failed to install packages: {e}")
        return False

def build_features():
    """Run the build_multi_features.py script"""
    print("\n🔨 Building multi-features JSON with tags...")
    data_dir = Path("data")
    if not data_dir.exists():
        print(f"✗ Data directory not found at {data_dir.absolute()}")
        return False
    
    try:
        os.chdir("data")
        subprocess.check_call([sys.executable, "build_multi_features.py"])
        os.chdir("..")
        print("✓ Multi-features JSON created successfully!")
        return True
    except Exception as e:
        os.chdir("..")
        print(f"✗ Failed to build features: {e}")
        return False

def create_index():
    """Create Elasticsearch index"""
    print("\n📊 Creating Elasticsearch index...")
    data_dir = Path("data")
    try:
        os.chdir("data")
        subprocess.check_call([sys.executable, "create_index.py"])
        os.chdir("..")
        print("✓ Index created successfully!")
        return True
    except Exception as e:
        os.chdir("..")
        print(f"✗ Failed to create index: {e}")
        return False

def bulk_index():
    """Bulk index data into Elasticsearch"""
    print("\n📤 Bulk indexing data into Elasticsearch...")
    data_dir = Path("data")
    try:
        os.chdir("data")
        subprocess.check_call([sys.executable, "bulk_index_multi.py"])
        os.chdir("..")
        print("✓ Data indexed successfully!")
        return True
    except Exception as e:
        os.chdir("..")
        print(f"✗ Failed to index data: {e}")
        return False

def main():
    print("=" * 60)
    print("Text-Image Search Engine - Elasticsearch Setup")
    print("=" * 60)
    
    # Step 1: Check Elasticsearch
    print("\n[Step 1/5] Checking Elasticsearch connection...")
    if not check_elasticsearch():
        print("\n⚠️  Elasticsearch is not running!")
        print("\nTo start Elasticsearch:")
        print("  1. Download from: https://www.elastic.co/downloads/elasticsearch")
        print("  2. Extract the archive")
        print("  3. Run: .\\bin\\elasticsearch.bat (Windows)")
        print("  4. Wait for it to start (usually takes 30-60 seconds)")
        print("  5. Re-run this script")
        return
    
    # Step 2: Install dependencies
    print("\n[Step 2/5] Installing dependencies...")
    if not install_dependencies():
        print("\n⚠️  Failed to install dependencies. Please install manually:")
        print("  pip install elasticsearch numpy requests")
        return
    
    # Step 3: Build features
    print("\n[Step 3/5] Building feature vectors with tags...")
    if not build_features():
        print("\n⚠️  Failed to build features. Please check your data folder structure.")
        return
    
    # Step 4: Create index
    print("\n[Step 4/5] Creating Elasticsearch index...")
    if not create_index():
        print("\n⚠️  Failed to create index.")
        return
    
    # Step 5: Bulk index
    print("\n[Step 5/5] Indexing data...")
    if not bulk_index():
        print("\n⚠️  Failed to index data.")
        return
    
    print("\n" + "=" * 60)
    print("✓ Setup completed successfully!")
    print("=" * 60)
    print("\nYou can now run the backend server:")
    print("  cd backend")
    print("  uvicorn main:app --reload")
    print("\nAnd the frontend:")
    print("  cd frontend")
    print("  streamlit run streamlit_app.py")

if __name__ == "__main__":
    main()
