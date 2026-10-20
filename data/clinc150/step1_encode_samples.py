"""Step 1: Encode CLINC150 text samples with a BART encoder + PCA to 64 dims.

Input:  HuggingFace dataset cmaldona/All-Generalization-OOD-CLINC150 (train split)
Output: samples_embedded.pkl
            {"embeddings": torch.Tensor (15200, 64), "pca": PCA object, "target_dim": 64}

Run from data/Clinc150/:
    python step1_encode_samples.py
"""

import pickle as pkl
from pathlib import Path

import pandas as pd
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer

DATA_DIR = Path(".")
TARGET_DIM = 64


class BARTEncoder:
    def __init__(self, device=None, target_dim=TARGET_DIM):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        self.model.to(self.device).eval()
        self.pca = PCA(n_components=target_dim)
        self.pca_fitted = False
        self.target_dim = target_dim

    def encode(self, texts, apply_pca=True):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True,
                                truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            z = self.model.model.encoder(**inputs).last_hidden_state.mean(dim=1).cpu()
        if apply_pca:
            z_np = z.numpy()
            if not self.pca_fitted:
                z_np = self.pca.fit_transform(z_np)
                self.pca_fitted = True
                print(f"PCA fitted: 768 -> {self.target_dim} dims, "
                      f"explained variance: {self.pca.explained_variance_ratio_.sum():.3f}")
            else:
                z_np = self.pca.transform(z_np)
            z = torch.from_numpy(z_np)
        return z

    def encode_batch(self, texts, batch_size=64):
        parts = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            parts.append(self.encode(texts[i:i + batch_size]))
        return torch.cat(parts, dim=0)


samples_df = pd.read_csv(
    "hf://datasets/cmaldona/All-Generalization-OOD-CLINC150/train_clinc150.csv"
)
print(f"Loaded {len(samples_df)} samples")

encoder = BARTEncoder(target_dim=TARGET_DIM)
embeddings = encoder.encode_batch(samples_df["data"].tolist(), batch_size=64)

pkl.dump({"embeddings": embeddings, "pca": encoder.pca, "target_dim": TARGET_DIM},
         open(DATA_DIR / "samples_embedded.pkl", "wb"))
print(f"Saved samples_embedded.pkl: {embeddings.shape}")
