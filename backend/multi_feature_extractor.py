import base64
import numpy as np
import cv2
from skimage import io
from sklearn.preprocessing import normalize
import os
import json

class MultiFeatureExtractor:
    """
    Feature extractor that can extract multiple types of features
    compatible with the indexed data (edge_histogram, gist, homogeneous_texture, surf)
    """
    
    def __init__(self):
        """Initialize the multi-feature extractor"""
        pass
    
    def extract_edge_histogram(self, img):
        """
        Extract edge histogram features (simplified version)
        Returns: 150-dimensional vector
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # Apply Sobel edge detection
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate edge magnitude and direction
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        direction = np.arctan2(sobel_y, sobel_x)
        
        # Create histogram of edge directions
        hist, _ = np.histogram(direction.flatten(), bins=150, range=(-np.pi, np.pi))
        
        # Normalize
        hist = hist.astype(np.float32)
        if np.sum(hist) > 0:
            hist = hist / np.sum(hist)
            
        return hist
    
    def extract_gist(self, img):
        """
        Extract GIST-like features (simplified version)
        Returns: 480-dimensional vector
        """
        # Resize image to standard size
        img_resized = cv2.resize(img, (64, 64))
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY) if len(img_resized.shape) == 3 else img_resized
        
        # Apply Gabor filters at different orientations and scales
        features = []
        
        # Multiple orientations and frequencies
        orientations = [0, 45, 90, 135]  # degrees
        frequencies = [0.1, 0.3, 0.5]
        
        for freq in frequencies:
            for angle in orientations:
                # Create Gabor kernel
                kernel = cv2.getGaborKernel((21, 21), 5, np.radians(angle), 2*np.pi*freq, 0.5, 0, ktype=cv2.CV_32F)
                
                # Apply filter
                filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
                
                # Divide image into grid and compute mean response
                h, w = filtered.shape
                grid_h, grid_w = 4, 4  # 4x4 grid
                cell_h, cell_w = h // grid_h, w // grid_w
                
                for i in range(grid_h):
                    for j in range(grid_w):
                        cell = filtered[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                        features.append(np.mean(cell))
        
        # Pad or truncate to 480 dimensions
        features = np.array(features)
        if len(features) < 480:
            features = np.pad(features, (0, 480 - len(features)), 'constant')
        else:
            features = features[:480]
            
        # Normalize
        features = features.astype(np.float32)
        if np.linalg.norm(features) > 0:
            features = features / np.linalg.norm(features)
            
        return features
    
    def extract_homogeneous_texture(self, img):
        """
        Extract homogeneous texture features (simplified version)
        Returns: 43-dimensional vector
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # Resize for consistency
        gray = cv2.resize(gray, (128, 128))
        
        # Calculate texture features using co-occurrence matrix approach
        features = []
        
        # Local Binary Pattern like features
        for offset in [(1, 0), (0, 1), (1, 1), (-1, 1)]:
            dx, dy = offset
            
            # Shift image
            shifted = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
            
            # Calculate differences
            diff = gray.astype(np.int16) - shifted.astype(np.int16)
            
            # Histogram of differences
            hist, _ = np.histogram(diff, bins=10, range=(-255, 255))
            features.extend(hist)
        
        # Add some statistical features
        features.extend([
            np.mean(gray),
            np.std(gray),
            np.min(gray),
            np.max(gray)
        ])
        
        # Ensure exactly 43 features
        features = np.array(features[:43])
        if len(features) < 43:
            features = np.pad(features, (0, 43 - len(features)), 'constant')
            
        # Normalize
        features = features.astype(np.float32)
        if np.linalg.norm(features) > 0:
            features = features / np.linalg.norm(features)
            
        return features
    
    def extract_surf_like(self, img):
        """
        Extract SURF-like features (simplified version)
        Returns: 128-dimensional vector (compatible with surf_vector field)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # Use ORB as a substitute for SURF (SURF is patented)
        orb = cv2.ORB_create(nfeatures=50)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        
        if descriptors is not None:
            # Take mean of descriptors to get 32-dimensional base features
            base_features = np.mean(descriptors, axis=0)
            
            # Pad to 128 dimensions to match the surf_vector mapping
            features = np.zeros(128)
            features[:len(base_features)] = base_features
        else:
            # Fallback to 128-dimensional zero vector
            features = np.zeros(128)
            
        return features.astype(np.float32)
    
    def extract_all_features(self, img):
        """
        Extract all feature types from an image
        Returns: dictionary with all feature types
        """
        features = {
            'edge_histogram': self.extract_edge_histogram(img),
            'gist': self.extract_gist(img),
            'homogeneous_texture': self.extract_homogeneous_texture(img),
            'surf': self.extract_surf_like(img)
        }
        return features
    
    def get_from_image(self, img_base64):
        """
        Extract features from base64 encoded image
        """
        image_bytes = base64.b64decode(img_base64)
        image_np = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        return self.extract_all_features(img)
    
    def get_from_link(self, img_link):
        """
        Extract features from image URL
        """
        image = io.imread(img_link)
        if len(image.shape) == 3 and image.shape[2] == 4:
            # Convert RGBA to RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        return self.extract_all_features(image)
    
    def get_from_file(self, img_path):
        """
        Extract features from local image file
        """
        img = cv2.imread(img_path)
        return self.extract_all_features(img)

class CombinedFeatureExtractor:
    """
    Creates a combined feature vector from the 4 feature types
    """
    
    def __init__(self):
        self.multi_extractor = MultiFeatureExtractor()
        # Weights for combining features
        self.weights = {
            'edge_histogram': 0.25,
            'gist': 0.40,  # Give more weight to GIST as it's more comprehensive
            'homogeneous_texture': 0.25,
            'surf': 0.10  # Lower weight as SURF is simplified
        }
    
    def combine_features(self, features_dict):
        """
        Combine the 4 feature types into a single vector
        """
        combined = []
        
        # Combine features with weights
        for feature_name, weight in self.weights.items():
            if feature_name in features_dict:
                weighted_features = features_dict[feature_name] * weight
                combined.extend(weighted_features)
        
        combined = np.array(combined)
        
        # Normalize the combined vector
        if np.linalg.norm(combined) > 0:
            combined = combined / np.linalg.norm(combined)
            
        return combined
    
    def get_from_image(self, img_base64):
        """Extract and combine features from base64 image"""
        features_dict = self.multi_extractor.get_from_image(img_base64)
        return self.combine_features(features_dict)
    
    def get_from_link(self, img_link):
        """Extract and combine features from image URL"""
        features_dict = self.multi_extractor.get_from_link(img_link)
        return self.combine_features(features_dict)