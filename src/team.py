from typing import Generator, Iterable, List, TypeVar
from collections import defaultdict
import os

import numpy as np
import supervision as sv
import torch
import umap
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import AutoProcessor, SiglipVisionModel
from huggingface_hub import login

V = TypeVar("V")

SIGLIP_MODEL_PATH = "google/siglip-base-patch16-224"


def setup_hf_authentication():
    """
    Setup Hugging Face authentication for Google Colab and other environments.
    
    This function checks for HF token in multiple ways:
    1. Environment variable HF_TOKEN
    2. Google Colab userdata (for Colab environments)
    3. Manual input prompt
    """
    hf_token = None
    
    # Try to get token from environment variable
    hf_token = os.getenv('HF_TOKEN')
    
    # If not found and running in Colab, try to get from userdata
    if hf_token is None:
        try:
            from google.colab import userdata
            hf_token = userdata.get('HF_TOKEN')
            print("✅ Found HF token in Colab userdata")
        except ImportError:
            # Not in Colab environment
            pass
        except Exception as e:
            print(f"⚠️ Could not access Colab userdata: {e}")
    
    # If still not found, check if we can access the model without token
    if hf_token is None:
        print("ℹ️ No HF token found, trying to access model without authentication...")
        return False
    
    # Login with the token
    try:
        login(token=hf_token, add_to_git_credential=True)
        print("✅ Successfully authenticated with Hugging Face")
        return True
    except Exception as e:
        print(f"❌ Failed to authenticate with Hugging Face: {e}")
        return False


def create_batches(
    sequence: Iterable[V], batch_size: int
) -> Generator[List[V], None, None]:
    """
    Generate batches from a sequence with a specified batch size.

    Args:
        sequence (Iterable[V]): The input sequence to be batched.
        batch_size (int): The size of each batch.

    Yields:
        Generator[List[V], None, None]: A generator yielding batches of the input
            sequence.
    """
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch


class TeamClassifier:
    """
    A classifier that uses a pre-trained SiglipVisionModel for feature extraction,
    UMAP for dimensionality reduction, and KMeans for clustering.
    """

    def __init__(self, device: str = "cpu", batch_size: int = 32, use_auth: bool = False):
        """
        Initialize the TeamClassifier with device and batch size.

        Args:
            device (str): The device to run the model on ('cpu' or 'cuda').
            batch_size (int): The batch size for processing images.
            use_auth (bool): Whether to attempt HF authentication (useful for Colab).
        """
        self.device = device
        self.batch_size = batch_size
        
        # Setup HF authentication if requested
        if use_auth:
            setup_hf_authentication()
        
        # Load the model and processor
        try:
            print(f"🔄 Loading SigLIP model: {SIGLIP_MODEL_PATH}")
            self.features_model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH).to(device)
            self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
            print("✅ SigLIP model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load SigLIP model: {e}")
            print("💡 If you're in Colab, make sure to set your HF_TOKEN in Colab secrets")
            print("   Go to: Runtime → Manage Sessions → Secrets → Add HF_TOKEN")
            raise
            
        self.reducer = umap.UMAP(n_components=3)
        self.cluster_model = KMeans(n_clusters=2)

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from a list of image crops using the pre-trained
            SiglipVisionModel.

        Args:
            crops (List[np.ndarray]): List of image crops.

        Returns:
            np.ndarray: Extracted features as a numpy array.
        """
        crops = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = create_batches(crops, self.batch_size)
        data = []
        with torch.no_grad():
            for batch in tqdm(batches, desc="Embedding extraction"):
                inputs = self.processor(images=batch, return_tensors="pt").to(
                    self.device
                )
                outputs = self.features_model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)

        return np.concatenate(data)

    def fit(self, crops: List[np.ndarray]) -> None:
        """
        Fit the classifier model on a list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.
        """
        data = self.extract_features(crops)
        projections = self.reducer.fit_transform(data)
        self.cluster_model.fit(projections)

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Predict the cluster labels for a list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.

        Returns:
            np.ndarray: Predicted cluster labels.
        """
        if len(crops) == 0:
            return np.array([])

        data = self.extract_features(crops)
        projections = self.reducer.transform(data)
        return self.cluster_model.predict(projections)


class TeamConsistencyTracker:
    def __init__(self, history_length=10, confidence_threshold=0.7):
        self.history_length = history_length
        self.confidence_threshold = confidence_threshold
        self.player_team_history = defaultdict(list)
        self.team_assignments = {}

    def update_team_assignment(self, tracker_id, current_team_prediction):
        # Add current prediction to history
        self.player_team_history[tracker_id].append(current_team_prediction)

        # Keep only recent history
        if len(self.player_team_history[tracker_id]) > self.history_length:
            self.player_team_history[tracker_id].pop(0)

        # Get most common team in history
        history = self.player_team_history[tracker_id]
        if len(history) >= 3:  # Wait for some history to accumulate
            team_counts = np.bincount(history)   # Count occurrences: [team_0_count, team_1_count]
            most_common_team = np.argmax(team_counts) # Get team with most votes
            confidence = team_counts[most_common_team] / len(history) # Calculate confidence

            if confidence >= self.confidence_threshold:
                self.team_assignments[tracker_id] = most_common_team
                return most_common_team

        # If not enough confidence, use current prediction
        return current_team_prediction

    def get_stable_team(self, tracker_id, current_team_prediction):
        if tracker_id in self.team_assignments:
            return self.team_assignments[tracker_id]
        return current_team_prediction
