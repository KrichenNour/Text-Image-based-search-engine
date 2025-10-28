import backend_config
import backend_config as config
from elasticsearch import Elasticsearch
from feature_extractor import FeatureExtractor
from multi_feature_extractor import MultiFeatureExtractor

client = Elasticsearch(
    [backend_config.elastic_url],
    basic_auth=(config.elastic_usr, config.elastic_pass),
    verify_certs=False
)
fe = FeatureExtractor()  # Original MobileNet extractor
multi_fe = MultiFeatureExtractor()  # New multi-feature extractor


def get_results_search_by_text(query, search_type, show_result):
    """Search by text tags (unchanged)"""
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
    return {"resulttype": results}


def get_results_search_by_image_single_feature(img, show_result, feature_field="gist"):
    """
    Search using a single feature type
    Available feature_fields: 'edge_histogram', 'gist', 'homogeneous_texture'
    """
    # Extract features using the multi-feature extractor
    features_dict = multi_fe.get_from_image(img)
    
    if feature_field not in features_dict:
        raise ValueError(f"Feature field '{feature_field}' not available")
    
    query_vector = features_dict[feature_field].tolist()
    
    body = {
        "knn": {
            "field": feature_field,
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


def get_results_search_by_image_multi_feature(img, show_result, feature_weights=None):
    """
    Search using multiple feature types with weighted scoring
    """
    if feature_weights is None:
        feature_weights = {
            'edge_histogram': 0.2,
            'gist': 0.5,
            'homogeneous_texture': 0.3
        }
    
    # Extract features using the multi-feature extractor
    features_dict = multi_fe.get_from_image(img)
    
    # Create multiple KNN queries
    knn_queries = []
    for feature_field, weight in feature_weights.items():
        if feature_field in features_dict:
            query_vector = features_dict[feature_field].tolist()
            knn_queries.append({
                "field": feature_field,
                "query_vector": query_vector,
                "k": show_result * 2,  # Get more candidates for each feature
                "num_candidates": max(100, show_result * 10),
                "boost": weight
            })
    
    # Use the first feature as primary and others as secondary
    if len(knn_queries) > 0:
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
    else:
        # Fallback to single feature
        return get_results_search_by_image_single_feature(img, show_result, "gist")


def get_results_search_by_image(img, show_result):
    """
    Main image search function - uses multi-feature search by default
    """
    return get_results_search_by_image_multi_feature(img, show_result)


def get_results_search_by_url(url, show_result):
    """
    Search by image URL using multi-feature approach
    """
    # Extract features using the multi-feature extractor
    features_dict = multi_fe.get_from_link(url)
    
    # Use multi-feature search with default weights
    feature_weights = {
        'edge_histogram': 0.2,
        'gist': 0.5,
        'homogeneous_texture': 0.3
    }
    
    # Create multiple KNN queries
    knn_queries = []
    for feature_field, weight in feature_weights.items():
        if feature_field in features_dict:
            query_vector = features_dict[feature_field].tolist()
            knn_queries.append({
                "field": feature_field,
                "query_vector": query_vector,
                "k": show_result * 2,
                "num_candidates": max(100, show_result * 10),
                "boost": weight
            })
    
    if len(knn_queries) > 0:
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
    else:
        # Fallback to gist feature only
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


def get_results_search_by_image_and_text(img, query, search_type, show_result):
    """
    Combined image and text search using multi-feature approach
    """
    # Extract features using the multi-feature extractor
    features_dict = multi_fe.get_from_image(img)
    
    # Feature weights for image similarity
    feature_weights = {
        'edge_histogram': 0.2,
        'gist': 0.5,
        'homogeneous_texture': 0.3
    }
    
    # Create multiple KNN queries
    knn_queries = []
    for feature_field, weight in feature_weights.items():
        if feature_field in features_dict:
            query_vector = features_dict[feature_field].tolist()
            knn_queries.append({
                "field": feature_field,
                "query_vector": query_vector,
                "k": show_result * 2,
                "num_candidates": max(100, show_result * 10),
                "boost": weight * 0.7  # Reduce image weight when combined with text
            })
    
    # Create text query
    if search_type == "multi_match":
        text_query = {
            "match": {
                "Tags": {
                    "query": query,
                    "boost": 0.3  # Text gets 30% weight
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
    
    results = client.search(
        index=backend_config.index_name, 
        body=body,
        size=show_result
    )
    return {"resulttype": results}


def get_results_search_by_feature_type(img, feature_type, show_result):
    """
    Search using a specific feature type only
    Useful for comparing different feature types
    """
    available_features = ['edge_histogram', 'gist', 'homogeneous_texture']
    
    if feature_type not in available_features:
        raise ValueError(f"Feature type must be one of: {available_features}")
    
    return get_results_search_by_image_single_feature(img, show_result, feature_type)