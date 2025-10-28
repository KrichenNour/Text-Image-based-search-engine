import backend_config
import backend_config as config
from elasticsearch import Elasticsearch
from multi_feature_extractor import MultiFeatureExtractor
from modern_feature_extractor import ModernFeatureExtractor
import numpy as np

client = Elasticsearch(
    [backend_config.elastic_url],
    basic_auth=(config.elastic_usr, config.elastic_pass),
    verify_certs=False,
    request_timeout=60,  # Increase timeout to 60 seconds for large KNN searches
    max_retries=3,
    retry_on_timeout=True
)
multi_fe = MultiFeatureExtractor()  # Use the new multi-feature extractor
modern_fe = ModernFeatureExtractor(use_gpu=True)  # Modern ResNet50 feature extractor with GPU support


def get_results_search_by_text(query, search_type, show_result):
    if search_type == "match":
        search_body = {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"Tags": query}},
                    ]
                }
            },
            "_source": ["OriginalURL", "ImageID", "Tags"],
        }
    elif search_type == "fuzzy":
        search_body = {
            "query": {"fuzzy": {"Tags": {"value": query}}},
            "_source": ["OriginalURL", "ImageID", "Tags"],
        }
    results = client.search(
        index=backend_config.index_name, body=search_body, size=show_result
    )
    print(f"DEBUG: Requested {show_result} results, got {len(results['hits']['hits'])} hits, total={results['hits']['total']['value']}")
    # Convert Elasticsearch response to dict to ensure proper serialization
    results_dict = results.body if hasattr(results, 'body') else dict(results)
    return {"resulttype": results_dict}


def get_results_search_by_url(url, show_result):
    """
    Search by image URL using modern ResNet50 embeddings
    Falls back to multi-feature search if modern embeddings are not available
    """
    try:
        # Extract modern ResNet50 features
        resnet_embedding = modern_fe.get_from_link(url)
        
        # Check if vector has zero magnitude
        magnitude = np.linalg.norm(resnet_embedding)
        if magnitude < 1e-6:
            print("Warning: ResNet embedding has zero magnitude, falling back to multi-feature search")
            return get_results_search_by_url_multifeature(url, show_result)
        
        # Normalize for cosine similarity
        query_vector = (resnet_embedding / magnitude).tolist()
        
        # Use KNN search with ResNet embeddings
        body = {
            "knn": {
                "field": "resnet_embedding",
                "query_vector": query_vector,
                "k": show_result,
                "num_candidates": max(100, show_result * 10)
            },
            "_source": ["OriginalURL", "ImageID", "Tags"],
        }
        
        results = client.search(
            index=backend_config.index_name, 
            body=body,
            size=show_result
        )
        print(f"DEBUG (URL search with ResNet): Requested {show_result} results, got {len(results['hits']['hits'])} hits, total={results['hits']['total']['value']}")
        
        # Convert Elasticsearch response to dict
        results_dict = results.body if hasattr(results, 'body') else dict(results)
        return {"resulttype": results_dict}
        
    except Exception as e:
        print(f"Error in modern feature search: {e}")
        # Fall back to multi-feature search
        return get_results_search_by_url_multifeature(url, show_result)


def get_results_search_by_url_multifeature(url, show_result):
    """
    Legacy multi-feature search (fallback method)
    """
    # Extract all feature types using the multi-feature extractor
    features_dict = multi_fe.get_from_link(url)
    
    # Use all 4 features with equal weight (0.25 each)
    knn_queries = []
    feature_weights = {
        'edge_histogram': 0.25,
        'gist': 0.25,
        'homogeneous_texture': 0.25,
        'surf_vector': 0.25  # Using the new searchable surf_vector field
    }
    
    import numpy as np
    for feature_type, weight in feature_weights.items():
        if feature_type in features_dict or (feature_type == 'surf_vector' and 'surf' in features_dict):
            # Map surf to surf_vector for the new searchable field
            if feature_type == 'surf_vector':
                query_vector = features_dict['surf']
            else:
                query_vector = features_dict[feature_type]
            
            # Check if vector has zero magnitude (to avoid cosine similarity error)
            magnitude = np.linalg.norm(query_vector)
            if magnitude < 1e-6:
                # Skip this feature if it's a zero vector
                print(f"Warning: {feature_type} has zero magnitude, skipping")
                continue
            
            # Ensure vector is normalized for cosine similarity
            query_vector = (query_vector / magnitude).tolist()
                
            knn_queries.append({
                "field": feature_type,
                "query_vector": query_vector,
                "k": show_result * 2,  # Get more candidates for better merging
                "num_candidates": max(100, show_result * 10),
                "boost": weight
            })
    
    # Check if we have valid queries
    if not knn_queries:
        print("Warning: No valid feature vectors found for search")
        return {"resulttype": {"hits": {"hits": [], "total": {"value": 0}}}}
    
    # Use multi-KNN search
    body = {
        "knn": knn_queries,
        "_source": ["OriginalURL", "ImageID", "Tags"],
    }
    
    results = client.search(
        index=backend_config.index_name, 
        body=body,
        size=show_result
    )
    print(f"DEBUG (URL search multifeature): Requested {show_result} results, got {len(results['hits']['hits'])} hits, total={results['hits']['total']['value']}")
    # Convert Elasticsearch response to dict to ensure proper serialization
    results_dict = results.body if hasattr(results, 'body') else dict(results)
    return {"resulttype": results_dict}


def get_results_search_by_image(img, show_result):
    """
    Search by uploaded image using modern ResNet50 embeddings
    Falls back to multi-feature search if modern embeddings are not available
    """
    try:
        # Extract modern ResNet50 features
        resnet_embedding = modern_fe.get_from_image(img)
        
        # Check if vector has zero magnitude
        magnitude = np.linalg.norm(resnet_embedding)
        if magnitude < 1e-6:
            print("Warning: ResNet embedding has zero magnitude, falling back to multi-feature search")
            return get_results_search_by_image_multifeature(img, show_result)
        
        # Normalize for cosine similarity
        query_vector = (resnet_embedding / magnitude).tolist()
        
        # Use KNN search with ResNet embeddings
        body = {
            "knn": {
                "field": "resnet_embedding",
                "query_vector": query_vector,
                "k": show_result,
                "num_candidates": max(100, show_result * 10)
            },
            "_source": ["OriginalURL", "ImageID", "Tags"],
        }
        
        results = client.search(
            index=backend_config.index_name, 
            body=body,
            size=show_result
        )
        print(f"DEBUG (Image search with ResNet): Requested {show_result} results, got {len(results['hits']['hits'])} hits, total={results['hits']['total']['value']}")
        
        # Convert Elasticsearch response to dict
        results_dict = results.body if hasattr(results, 'body') else dict(results)
        return {"resulttype": results_dict}
        
    except Exception as e:
        print(f"Error in modern feature search: {e}")
        import traceback
        traceback.print_exc()
        # Fall back to multi-feature search
        return get_results_search_by_image_multifeature(img, show_result)


def get_results_search_by_image_multifeature(img, show_result):
    """
    Legacy multi-feature search (fallback method)
    """
    # Extract all feature types using the multi-feature extractor
    features_dict = multi_fe.get_from_image(img)
    
    # Use all 4 features with equal weight (0.25 each)
    knn_queries = []
    feature_weights = {
        'edge_histogram': 0.25,
        'gist': 0.25,
        'homogeneous_texture': 0.25,
        'surf_vector': 0.25  # Using the new searchable surf_vector field
    }
    
    import numpy as np
    for feature_type, weight in feature_weights.items():
        if feature_type in features_dict or (feature_type == 'surf_vector' and 'surf' in features_dict):
            # Map surf to surf_vector for the new searchable field
            if feature_type == 'surf_vector':
                query_vector = features_dict['surf']
            else:
                query_vector = features_dict[feature_type]
            
            # Check if vector has zero magnitude (to avoid cosine similarity error)
            magnitude = np.linalg.norm(query_vector)
            if magnitude < 1e-6:
                # Skip this feature if it's a zero vector
                print(f"Warning: {feature_type} has zero magnitude, skipping")
                continue
            
            # Ensure vector is normalized for cosine similarity
            query_vector = (query_vector / magnitude).tolist()
                
            knn_queries.append({
                "field": feature_type,
                "query_vector": query_vector,
                "k": show_result * 2,  # Get more candidates for better merging
                "num_candidates": max(100, show_result * 10),
                "boost": weight
            })
    
    # Check if we have valid queries
    if not knn_queries:
        print("Warning: No valid feature vectors found for search")
        return {"resulttype": {"hits": {"hits": [], "total": {"value": 0}}}}
    
    # Use multi-KNN search
    body = {
        "knn": knn_queries,
        "_source": ["OriginalURL", "ImageID", "Tags"],
    }
    
    results = client.search(
        index=backend_config.index_name, 
        body=body,
        size=show_result
    )
    print(f"DEBUG (Image search multifeature): Requested {show_result} results, got {len(results['hits']['hits'])} hits, total={results['hits']['total']['value']}")
    # Convert Elasticsearch response to dict to ensure proper serialization
    results_dict = results.body if hasattr(results, 'body') else dict(results)
    return {"resulttype": results_dict}


def get_results_search_by_image_and_text(img, query, search_type, show_result):
    """
    Combined search: Image (ResNet50) + Text (Tags)
    Uses 70% weight for image similarity, 30% for text matching
    """
    try:
        # Extract ResNet50 features from image
        resnet_embedding = modern_fe.get_from_image(img)
        
        # Check if vector has zero magnitude
        magnitude = np.linalg.norm(resnet_embedding)
        if magnitude < 1e-6:
            print("Warning: ResNet embedding has zero magnitude, using text-only search")
            return get_results_search_by_text(query, search_type, show_result)
        
        # Normalize for cosine similarity
        query_vector = (resnet_embedding / magnitude).tolist()
        
        # Build combined query: KNN (image) + text match
        if search_type == "match" or search_type == "multi_match":
            body = {
                "query": {
                    "match": {
                        "Tags": {
                            "query": query,
                            "boost": 0.3  # Text gets 30% weight
                        }
                    }
                },
                "knn": {
                    "field": "resnet_embedding",
                    "query_vector": query_vector,
                    "k": show_result * 2,  # Get more candidates for better merging
                    "num_candidates": max(100, show_result * 10),
                    "boost": 0.7  # Image gets 70% weight
                },
                "_source": ["OriginalURL", "ImageID", "Tags"],
            }
        else:  # fuzzy
            body = {
                "query": {
                    "fuzzy": {
                        "Tags": {
                            "value": query,
                            "boost": 0.3
                        }
                    }
                },
                "knn": {
                    "field": "resnet_embedding",
                    "query_vector": query_vector,
                    "k": show_result * 2,
                    "num_candidates": max(100, show_result * 10),
                    "boost": 0.7
                },
                "_source": ["OriginalURL", "ImageID", "Tags"],
            }
        
        results = client.search(
            index=backend_config.index_name, 
            body=body,
            size=show_result
        )
        print(f"DEBUG (Image+Text search with ResNet): Requested {show_result} results, got {len(results['hits']['hits'])} hits")
        
        # Convert Elasticsearch response to dict
        results_dict = results.body if hasattr(results, 'body') else dict(results)
        return {"resulttype": results_dict}
        
    except Exception as e:
        print(f"Error in combined search: {e}")
        import traceback
        traceback.print_exc()
        # Fall back to text-only search
        return get_results_search_by_text(query, search_type, show_result)


def get_index_count():
    """Returns the number of documents in the Elasticsearch index."""
    try:
        count = client.count(index=backend_config.index_name)
        return {"doc_count": count['count']}
    except Exception as e:
        print(f"Error getting index count: {e}")
        return {"doc_count": 0}
