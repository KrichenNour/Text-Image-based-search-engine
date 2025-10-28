"""
Advanced Multi-Feature Search Functions
Provides multiple approaches for combining and using the 4 feature types
"""
import backend_config
import backend_config as config
from elasticsearch import Elasticsearch
from multi_feature_extractor import MultiFeatureExtractor, CombinedFeatureExtractor
import numpy as np

client = Elasticsearch(
    [backend_config.elastic_url],
    basic_auth=(config.elastic_usr, config.elastic_pass),
    verify_certs=False
)

multi_fe = MultiFeatureExtractor()
combined_fe = CombinedFeatureExtractor()


def search_by_single_feature(img, feature_type, show_result):
    """
    Search using only one specific feature type
    feature_type: 'edge_histogram', 'gist', or 'homogeneous_texture'
    """
    features_dict = multi_fe.get_from_image(img)
    
    if feature_type not in features_dict:
        raise ValueError(f"Feature type '{feature_type}' not available")
    
    query_vector = features_dict[feature_type].tolist()
    
    body = {
        "knn": {
            "field": feature_type,
            "query_vector": query_vector,
            "k": show_result,
            "num_candidates": max(50, show_result * 5)
        },
        "_source": ["OriginalURL", "ImageID", "Tags"],
    }
    
    results = client.search(
        index=backend_config.index_name, body=body
    )
    return {"resulttype": results}


def search_by_weighted_features(img, show_result, weights=None):
    """
    Search using multiple features with custom weights
    weights: dict like {'edge_histogram': 0.3, 'gist': 0.5, 'homogeneous_texture': 0.2}
    """
    if weights is None:
        weights = {
            'edge_histogram': 0.25,
            'gist': 0.50,
            'homogeneous_texture': 0.25
        }
    
    features_dict = multi_fe.get_from_image(img)
    
    # For Elasticsearch, we can use multiple knn queries with different boosts
    # This will search all features and combine scores
    knn_queries = []
    
    for feature_type, weight in weights.items():
        if feature_type in features_dict and feature_type != 'surf':  # surf is not searchable
            query_vector = features_dict[feature_type].tolist()
            knn_queries.append({
                "field": feature_type,
                "query_vector": query_vector,
                "k": show_result * 2,  # Get more candidates for better merging
                "num_candidates": max(100, show_result * 10),
                "boost": weight
            })
    
    if len(knn_queries) == 0:
        # Fallback to gist only
        return search_by_single_feature(img, 'gist', show_result)
    
    body = {
        "knn": knn_queries,
        "_source": ["OriginalURL", "ImageID", "Tags"],
    }
    
    results = client.search(
        index=backend_config.index_name,
        body=body,
        size=show_result
    )
    return {"resulttype": results}


def search_by_ensemble_voting(img, show_result, top_k_per_feature=10):
    """
    Ensemble approach: get top results from each feature and combine by voting
    """
    features_dict = multi_fe.get_from_image(img)
    
    searchable_features = ['edge_histogram', 'gist', 'homogeneous_texture']
    all_results = {}
    
    # Get results from each feature type
    for feature_type in searchable_features:
        if feature_type in features_dict:
            result = search_by_single_feature(img, feature_type, top_k_per_feature)
            hits = result['resulttype']['hits']['hits']
            
            for i, hit in enumerate(hits):
                image_id = hit['_source']['ImageID']
                score = hit['_score']
                
                if image_id not in all_results:
                    all_results[image_id] = {
                        'hit': hit,
                        'scores': [],
                        'rank_sum': 0,
                        'count': 0
                    }
                
                all_results[image_id]['scores'].append(score)
                all_results[image_id]['rank_sum'] += (i + 1)  # Rank starts from 1
                all_results[image_id]['count'] += 1
    
    # Calculate ensemble scores
    for image_id, data in all_results.items():
        # Combine scores using average rank and score
        avg_rank = data['rank_sum'] / data['count']
        avg_score = np.mean(data['scores'])
        
        # Favor images that appear in multiple feature searches
        ensemble_score = (avg_score * data['count']) / avg_rank
        data['ensemble_score'] = ensemble_score
    
    # Sort by ensemble score and take top results
    sorted_results = sorted(all_results.values(), 
                          key=lambda x: x['ensemble_score'], 
                          reverse=True)
    
    # Format results like Elasticsearch response
    hits = []
    for i, result in enumerate(sorted_results[:show_result]):
        hit = result['hit'].copy()
        hit['_score'] = result['ensemble_score']
        hits.append(hit)
    
    # Create mock Elasticsearch response
    response = {
        "took": 1,
        "hits": {
            "total": {"value": len(hits), "relation": "eq"},
            "hits": hits
        }
    }
    
    return {"resulttype": response}


def search_with_feature_comparison(img, show_result):
    """
    Returns results from all three searchable features for comparison
    """
    features_dict = multi_fe.get_from_image(img)
    
    results = {}
    searchable_features = ['edge_histogram', 'gist', 'homogeneous_texture']
    
    for feature_type in searchable_features:
        if feature_type in features_dict:
            result = search_by_single_feature(img, feature_type, show_result)
            results[feature_type] = result
    
    return results


# Update the main search functions to use the advanced methods
def get_results_search_by_image(img, show_result, method="weighted"):
    """
    Main image search function with multiple methods
    method: 'single_gist', 'weighted', 'ensemble', 'comparison'
    """
    if method == "single_gist":
        return search_by_single_feature(img, 'gist', show_result)
    elif method == "weighted":
        return search_by_weighted_features(img, show_result)
    elif method == "ensemble":
        return search_by_ensemble_voting(img, show_result)
    elif method == "comparison":
        return search_with_feature_comparison(img, show_result)
    else:
        # Default to weighted
        return search_by_weighted_features(img, show_result)


def get_results_search_by_url(url, show_result, method="weighted"):
    """
    URL search with multiple methods
    """
    # Convert URL to image data
    import urllib.request
    import base64
    from PIL import Image
    import io
    
    try:
        # Download image
        with urllib.request.urlopen(url) as response:
            img_data = response.read()
        
        # Convert to base64
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        
        # Use image search
        return get_results_search_by_image(img_base64, show_result, method)
        
    except Exception as e:
        # Fallback to direct feature extraction
        features_dict = multi_fe.get_from_link(url)
        query_vector = features_dict['gist'].tolist()
        
        body = {
            "knn": {
                "field": "gist",
                "query_vector": query_vector,
                "k": show_result,
                "num_candidates": max(50, show_result * 5),
            },
            "_source": ["OriginalURL", "ImageID", "Tags"],
        }
        results = client.search(
            index=backend_config.index_name, body=body, size=show_result
        )
        return {"resulttype": results}


def get_results_search_by_image_and_text(img, query, search_type, show_result, image_method="weighted"):
    """
    Combined image and text search
    """
    if image_method == "weighted":
        # Get weighted feature vectors
        features_dict = multi_fe.get_from_image(img)
        
        # Create weighted KNN queries
        knn_queries = []
        feature_weights = {
            'edge_histogram': 0.25,
            'gist': 0.50,
            'homogeneous_texture': 0.25
        }
        
        for feature_type, weight in feature_weights.items():
            if feature_type in features_dict:
                query_vector = features_dict[feature_type].tolist()
                knn_queries.append({
                    "field": feature_type,
                    "query_vector": query_vector,
                    "k": show_result * 2,
                    "num_candidates": max(100, show_result * 10),
                    "boost": weight * 0.7  # 70% weight for image
                })
        
        # Text query (30% weight)
        if search_type == "multi_match":
            text_query = {
                "match": {
                    "Tags": {
                        "query": query,
                        "boost": 0.3
                    }
                }
            }
        else:
            text_query = {
                "fuzzy": {
                    "Tags": {
                        "value": query,
                        "boost": 0.3
                    }
                }
            }
        
        body = {
            "query": text_query,
            "knn": knn_queries,
            "_source": ["OriginalURL", "ImageID", "Tags"],
        }
        
    else:
        # Use single feature (gist) for simplicity
        features_dict = multi_fe.get_from_image(img)
        image_vector = features_dict['gist'].tolist()
        
        if search_type == "multi_match":
            text_query = {
                "match": {
                    "Tags": {
                        "query": query,
                        "boost": 0.3
                    }
                }
            }
        else:
            text_query = {
                "fuzzy": {
                    "Tags": {
                        "value": query,
                        "boost": 0.3
                    }
                }
            }
        
        body = {
            "query": text_query,
            "knn": {
                "field": "gist",
                "query_vector": image_vector,
                "k": show_result,
                "num_candidates": max(50, show_result * 5),
                "boost": 0.7
            },
            "_source": ["OriginalURL", "ImageID", "Tags"],
        }
    
    results = client.search(
        index=backend_config.index_name, 
        body=body,
        size=show_result
    )
    return {"resulttype": results}