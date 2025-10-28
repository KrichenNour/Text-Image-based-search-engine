"""
Modern image feature extractor using deep learning models (ResNet, CLIP)
with GPU acceleration support
"""
import base64
import numpy as np
import cv2
from skimage import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import io as IO


class ModernFeatureExtractor:
    """
    Modern deep learning-based feature extractor with GPU support
    Uses ResNet50 pretrained on ImageNet for high-quality image embeddings
    """
    
    def __init__(self, use_gpu=True):
        """
        Initialize the feature extractor
        
        Args:
            use_gpu: Whether to use GPU if available
        """
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load ResNet50 pretrained model
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # Remove the final classification layer to get embeddings
        # ResNet50 outputs 2048-dimensional features before the FC layer
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Standard ImageNet preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print("ResNet50 model loaded successfully")
    
    def _extract_features(self, img):
        """
        Extract features from a PIL Image
        
        Args:
            img: PIL Image
            
        Returns:
            2048-dimensional feature vector (numpy array)
        """
        # Preprocess the image
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(img_tensor)
        
        # Flatten and convert to numpy
        features = features.squeeze().cpu().numpy()
        
        # Normalize the features
        features = features / (np.linalg.norm(features) + 1e-8)
        
        return features
    
    def extract_features_batch(self, image_paths, batch_size=64):
        """
        Extract features from multiple images in batches (TRUE GPU batching)
        
        Args:
            image_paths: List of image file paths
            batch_size: Number of images to process in each GPU batch
            
        Returns:
            List of 2048-dimensional feature vectors
        """
        all_features = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_tensors = []
            valid_indices = []
            
            # Load and preprocess all images in the batch
            for idx, img_path in enumerate(batch_paths):
                try:
                    img = Image.open(img_path)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img_tensor = self.transform(img)
                    batch_tensors.append(img_tensor)
                    valid_indices.append(i + idx)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    all_features.append(np.zeros(2048, dtype=np.float32))
                    continue
            
            if batch_tensors:
                # Stack tensors and process as a single batch on GPU
                batch_tensor = torch.stack(batch_tensors).to(self.device)
                
                with torch.no_grad():
                    features = self.model(batch_tensor)
                
                # Process each feature in the batch
                features = features.squeeze().cpu().numpy()
                if len(features.shape) == 1:  # Single image case
                    features = features.reshape(1, -1)
                
                for feat in features:
                    # Normalize
                    feat = feat / (np.linalg.norm(feat) + 1e-8)
                    all_features.append(feat)
        
        return all_features
    
    def get_from_image(self, img_base64):
        """
        Extract features from base64 encoded image
        
        Args:
            img_base64: Base64 encoded image string
            
        Returns:
            2048-dimensional feature vector
        """
        try:
            image_bytes = base64.b64decode(img_base64)
            img = Image.open(IO.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            return self._extract_features(img)
        except Exception as e:
            print(f"Error extracting features from image: {e}")
            # Return zero vector on error
            return np.zeros(2048, dtype=np.float32)
    
    def get_from_link(self, img_link):
        """
        Extract features from image URL
        
        Args:
            img_link: URL to the image
            
        Returns:
            2048-dimensional feature vector
        """
        try:
            # Load image from URL
            image = io.imread(img_link)
            
            # Convert to PIL Image
            if isinstance(image, np.ndarray):
                if len(image.shape) == 2:  # Grayscale
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[2] == 4:  # RGBA
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                img = Image.fromarray(image.astype('uint8'))
            else:
                img = image
            
            return self._extract_features(img)
        except Exception as e:
            print(f"Error extracting features from URL {img_link}: {e}")
            # Return zero vector on error
            return np.zeros(2048, dtype=np.float32)
    
    def get_from_file(self, img_path):
        """
        Extract features from local image file
        
        Args:
            img_path: Path to the image file
            
        Returns:
            2048-dimensional feature vector
        """
        try:
            img = Image.open(img_path)
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            return self._extract_features(img)
        except Exception as e:
            print(f"Error extracting features from file {img_path}: {e}")
            # Return zero vector on error
            return np.zeros(2048, dtype=np.float32)


class CLIPFeatureExtractor:
    """
    CLIP-based feature extractor (optional, more powerful but requires more memory)
    Only use if CLIP is installed: pip install git+https://github.com/openai/CLIP.git
    """
    
    def __init__(self, use_gpu=True):
        """
        Initialize CLIP feature extractor
        
        Args:
            use_gpu: Whether to use GPU if available
        """
        try:
            import clip
            self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
            print(f"Using device: {self.device}")
            
            # Load CLIP model (ViT-B/32 is a good balance of speed and quality)
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            
            print("CLIP model loaded successfully")
        except ImportError:
            raise ImportError(
                "CLIP is not installed. Install with: "
                "pip install git+https://github.com/openai/CLIP.git"
            )
    
    def _extract_features(self, img):
        """
        Extract features from a PIL Image
        
        Args:
            img: PIL Image
            
        Returns:
            512-dimensional feature vector (for ViT-B/32)
        """
        # Preprocess the image
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model.encode_image(img_tensor)
        
        # Convert to numpy and normalize
        features = features.squeeze().cpu().numpy()
        features = features / (np.linalg.norm(features) + 1e-8)
        
        return features
    
    def get_from_image(self, img_base64):
        """Extract features from base64 encoded image"""
        try:
            image_bytes = base64.b64decode(img_base64)
            img = Image.open(IO.BytesIO(image_bytes))
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            return self._extract_features(img)
        except Exception as e:
            print(f"Error extracting features from image: {e}")
            return np.zeros(512, dtype=np.float32)
    
    def get_from_link(self, img_link):
        """Extract features from image URL"""
        try:
            image = io.imread(img_link)
            
            if isinstance(image, np.ndarray):
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                img = Image.fromarray(image.astype('uint8'))
            else:
                img = image
            
            return self._extract_features(img)
        except Exception as e:
            print(f"Error extracting features from URL {img_link}: {e}")
            return np.zeros(512, dtype=np.float32)
    
    def get_from_file(self, img_path):
        """Extract features from local image file"""
        try:
            img = Image.open(img_path)
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            return self._extract_features(img)
        except Exception as e:
            print(f"Error extracting features from file {img_path}: {e}")
            return np.zeros(512, dtype=np.float32)


# Factory function to create the appropriate feature extractor
def create_feature_extractor(model_type='resnet', use_gpu=True):
    """
    Factory function to create a feature extractor
    
    Args:
        model_type: 'resnet' or 'clip'
        use_gpu: Whether to use GPU if available
        
    Returns:
        Feature extractor instance
    """
    if model_type == 'clip':
        return CLIPFeatureExtractor(use_gpu=use_gpu)
    else:
        return ModernFeatureExtractor(use_gpu=use_gpu)
