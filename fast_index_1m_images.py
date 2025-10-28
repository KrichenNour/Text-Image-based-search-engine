"""
Fast indexer for 1 million images with ResNet50 embeddings and tags only.
Optimized for speed with batch processing and GPU acceleration.
"""
import sys
import os
from pathlib import Path
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, parallel_bulk
import numpy as np
from tqdm import tqdm
import json
import argparse
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / 'backend'))

from modern_feature_extractor import ModernFeatureExtractor

# Configuration
ELASTIC_URL = "http://localhost:9200"
INDEX_NAME = "images_resnet"  # New optimized index
IMAGES_DIR = Path("data/images")
TAGS_DIR = Path("data/tags")
BATCH_SIZE = 768   # Balanced - not too large, not too small
GPU_BATCH_SIZE = 384  # Keep maxed - GPU loves this
BULK_SIZE = 10000  # Keep large ES batches
NUM_WORKERS = 16   # Keep parallel ES writes
IO_WORKERS = 12    # Reduced - 16 was causing contention


def load_tags(image_id, tags_dir):
    """Load tags from the tags folder structure"""
    # Tags are organized in folders 0-99
    for i in range(100):
        tag_path = tags_dir / str(i) / f"{image_id}.txt"
        if tag_path.exists():
            try:
                tags_text = tag_path.read_text(encoding="utf-8", errors="ignore").strip()
                return tags_text.replace("\n", " ")
            except Exception as e:
                print(f"Warning: Failed to load tags for {image_id}: {e}")
                return ""
    return ""


def find_image_file(image_id, images_dir):
    """Find the image file for a given image ID"""
    extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    
    # Images are organized in folders 0-99
    for folder in range(100):
        folder_path = images_dir / str(folder)
        if not folder_path.exists():
            continue
        
        for ext in extensions:
            img_path = folder_path / f"{image_id}{ext}"
            if img_path.exists():
                return img_path
    
    return None


def save_progress(processed_count):
    """Save indexing progress to file"""
    progress_file = Path(__file__).parent / "indexing_progress.txt"
    with open(progress_file, 'w') as f:
        f.write(str(processed_count))

def load_progress():
    """Load indexing progress from file"""
    progress_file = Path(__file__).parent / "indexing_progress.txt"
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                return int(f.read().strip())
        except:
            return None
    return None


def create_optimized_index(es, force_recreate=False):
    
    index_exists = es.indices.exists(index=INDEX_NAME)
    
    if force_recreate and index_exists:
        print(f"Deleting existing index '{INDEX_NAME}'...")
        es.indices.delete(index=INDEX_NAME)
        index_exists = False
    
    if not index_exists:
        mapping = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,  # No replicas for faster indexing
                "refresh_interval": "30s"  # Reduce refresh frequency during bulk indexing
            },
            "mappings": {
                "properties": {
                    "ImageID": {
                        "type": "keyword"
                    },
                    "OriginalURL": {
                        "type": "keyword",
                        "index": False  # Don't need to search by URL
                    },
                    "Tags": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "resnet_embedding": {
                        "type": "dense_vector",
                        "dims": 2048,
                        "index": True,
                        "similarity": "cosine"
                    }
                }
            }
        }
        
        es.indices.create(index=INDEX_NAME, body=mapping)
        print(f"Created optimized index '{INDEX_NAME}' (ResNet50 + Tags only)")
    else:
        print(f"Using existing index '{INDEX_NAME}'")


def get_all_image_ids(images_dir, max_images=None):
    """Get all image IDs from the images directory"""
    image_ids = []
    
    print("Scanning for images...")
    for folder in range(100):
        folder_path = images_dir / str(folder)
        if not folder_path.exists():
            continue
        
        for img_file in folder_path.glob("*"):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                image_ids.append(img_file.stem)
                
                if max_images and len(image_ids) >= max_images:
                    return image_ids
    
    print(f"Found {len(image_ids)} images")
    return image_ids


def extract_features_batch(image_paths, feature_extractor):
    """Extract ResNet50 features for a batch of images using TRUE GPU batching"""
    # Use the new batch method that processes multiple images on GPU simultaneously
    embeddings = feature_extractor.extract_features_batch(image_paths, batch_size=GPU_BATCH_SIZE)
    return embeddings


def generate_documents(image_ids, images_dir, tags_dir, feature_extractor):
    """Generator that yields documents ready for Elasticsearch"""
    
    total = len(image_ids)
    
    # Create thread pool for parallel I/O
    executor = ThreadPoolExecutor(max_workers=IO_WORKERS)
    
    for i in tqdm(range(0, total, BATCH_SIZE), desc="Extracting features"):
        batch_ids = image_ids[i:i+BATCH_SIZE]
        
        # Parallel image path finding
        def find_path(img_id):
            return find_image_file(img_id, images_dir)
        
        batch_paths = list(executor.map(find_path, batch_ids))
        valid_paths = [(p, batch_ids[idx]) for idx, p in enumerate(batch_paths) if p is not None]
        
        if not valid_paths:
            continue
        
        # Extract features in TRUE GPU batch
        paths_only = [p for p, _ in valid_paths]
        embeddings_list = extract_features_batch(paths_only, feature_extractor)
        
        # Parallel tag loading
        def load_tags_for_id(img_id):
            return load_tags(img_id, tags_dir)
        
        ids_only = [img_id for _, img_id in valid_paths]
        tags_list = list(executor.map(load_tags_for_id, ids_only))
        
        # Create documents (optimized - convert embeddings once)
        for (img_path, img_id), embedding, tags in zip(valid_paths, embeddings_list, tags_list):
            # Convert numpy array to list only once, reuse if possible
            if isinstance(embedding, np.ndarray):
                embedding_list = embedding.tolist()
            else:
                embedding_list = embedding
                
            doc = {
                "_index": INDEX_NAME,
                "_id": img_id,
                "_source": {
                    "ImageID": img_id,
                    "OriginalURL": f"https://example.com/{img_id}.jpg",  # Placeholder
                    "Tags": tags,
                    "resnet_embedding": embedding_list
                }
            }
            
            yield doc


def fast_index_images(use_gpu=True, max_images=None, resume_from=None):
    """
    Fast indexing of images with ResNet50 and tags only
    
    Args:
        use_gpu: Whether to use GPU acceleration
        max_images: Maximum number of images to index (None = all)
        resume_from: Resume from this image number (for interrupted runs)
    """
    print("=" * 70)
    print("Fast Indexer for 1M Images (ResNet50 + Tags only)")
    print("=" * 70)
    print()
    
    # Initialize
    es = Elasticsearch([ELASTIC_URL])
    
    # Check connection
    if not es.ping():
        print("ERROR: Cannot connect to Elasticsearch at", ELASTIC_URL)
        return False
    
    print("✓ Connected to Elasticsearch")
    
    # Auto-resume from progress file if no manual resume specified
    if resume_from is None:
        saved_progress = load_progress()
        if saved_progress is not None:
            resume_from = saved_progress
            print(f"Auto-resuming from saved progress: {resume_from}")
    
    # Create index (only recreate if not resuming)
    force_recreate = (resume_from is None)
    create_optimized_index(es, force_recreate=force_recreate)
    
    # Initialize feature extractor
    print("\nInitializing ResNet50 feature extractor...")
    feature_extractor = ModernFeatureExtractor(use_gpu=use_gpu)
    print(f"✓ Feature extractor ready (device: {feature_extractor.device})")
    
    # Get all image IDs
    image_ids = get_all_image_ids(IMAGES_DIR, max_images=max_images)
    
    if not image_ids:
        print("ERROR: No images found!")
        return False
    
    total_images = len(image_ids)
    print(f"\nTotal images to index: {total_images}")
    
    # Apply resume
    if resume_from:
        image_ids = image_ids[resume_from:]
        print(f"Resuming from image {resume_from}")
    
    # Disable refresh for faster bulk indexing
    es.indices.put_settings(
        index=INDEX_NAME,
        body={"index": {"refresh_interval": "-1"}}
    )
    
    # Start indexing
    print(f"\nStarting bulk indexing...")
    print(f"  Batch size: {BATCH_SIZE} (overall iteration)")
    print(f"  GPU batch size: {GPU_BATCH_SIZE} (true GPU batching)")
    print(f"  Bulk size: {BULK_SIZE} (Elasticsearch)")
    print(f"  Workers: {NUM_WORKERS} (parallel ES writes)")
    print()
    
    try:
        # Use parallel_bulk for faster indexing
        success_count = 0
        error_count = 0
        processed_images = 0
        
        for success, info in parallel_bulk(
            es,
            generate_documents(image_ids, IMAGES_DIR, TAGS_DIR, feature_extractor),
            thread_count=NUM_WORKERS,
            chunk_size=BULK_SIZE,
            raise_on_error=False
        ):
            if success:
                success_count += 1
                processed_images += BULK_SIZE  # Approximate
                # Save progress every 10 chunks
                if success_count % 10 == 0:
                    current_progress = (resume_from or 0) + processed_images
                    save_progress(current_progress)
            else:
                error_count += 1
                if error_count <= 10:  # Only print first 10 errors
                    print(f"Error: {info}")
        
        # Save final progress
        final_progress = (resume_from or 0) + processed_images
        save_progress(final_progress)
        
        print()
        print("=" * 70)
        print("Indexing Complete!")
        print("=" * 70)
        print(f"Successfully indexed: {success_count}")
        print(f"Errors: {error_count}")
        
        # Re-enable refresh and force refresh
        print("\nOptimizing index...")
        es.indices.put_settings(
            index=INDEX_NAME,
            body={"index": {"refresh_interval": "1s"}}
        )
        es.indices.refresh(index=INDEX_NAME)
        
        # Get final count
        final_count = es.count(index=INDEX_NAME)['count']
        print(f"Final document count: {final_count}")
        
        return True
        
    except Exception as e:
        print(f"\nERROR during indexing: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Always re-enable refresh
        try:
            es.indices.put_settings(
                index=INDEX_NAME,
                body={"index": {"refresh_interval": "1s"}}
            )
        except:
            pass


def estimate_time(num_images, use_gpu):
    """Estimate indexing time"""
    if use_gpu:
        seconds_per_image = 0.3  # ~300ms per image with GPU (including overhead)
    else:
        seconds_per_image = 2.0  # ~2s per image with CPU
    
    total_seconds = num_images * seconds_per_image
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    
    return hours, minutes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fast indexer for 1M images')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')
    parser.add_argument('--max-images', type=int, default=None, help='Maximum images to index')
    parser.add_argument('--resume-from', type=int, default=None, help='Resume from image number')
    parser.add_argument('--test', action='store_true', help='Test with 10 images only')
    
    args = parser.parse_args()
    
    use_gpu = not args.no_gpu
    max_images = 10 if args.test else args.max_images
    
    if max_images:
        hours, minutes = estimate_time(max_images, use_gpu)
        print(f"Estimated time for {max_images} images: {hours}h {minutes}m")
        print()
    
    success = fast_index_images(
        use_gpu=use_gpu,
        max_images=max_images,
        resume_from=args.resume_from
    )
    
    if success:
        print("\n✓ Indexing completed successfully!")
        print("\nNext steps:")
        print("  1. Update backend_config.py to use new index: 'images_resnet'")
        print("  2. Restart backend: cd backend && python main.py")
        print("  3. Test search: python test_modern_image_search.py")
    else:
        print("\n✗ Indexing failed!")
        sys.exit(1)
