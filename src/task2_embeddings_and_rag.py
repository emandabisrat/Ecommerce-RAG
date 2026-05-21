"""
Task 2 – Vision-Language RAG with CLIP + ChromaDB
==================================================
Objectives
----------
1. Load CLIP (ViT-B/32) via open_clip.
2. Generate text embeddings for all product descriptions.
3. Generate image embeddings where local images are available.
4. Store both sets of embeddings in ChromaDB collections.
5. Expose a unified `retrieve()` function supporting text, image, and hybrid queries.
6. Evaluation: Recall@1, Recall@5, Recall@10 on a held-out query set.
 
Run standalone:
    python -m src.task2_embeddings_and_rag
"""
 
import logging
from pathlib import Path
from typing import Optional, Union
import sys
 
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
import open_clip
import chromadb
from chromadb.config import Settings
 
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
 
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 1. CLIP model loader
# ─────────────────────────────────────────────────────────────────────────────
 
class CLIPEncoder:
    """
    Thin wrapper around open_clip for encoding text and images.
 
    CLIP (Contrastive Language–Image Pretraining) maps both modalities into
    the SAME 512-dim embedding space — enabling cross-modal similarity search.
    """
 
    def __init__(
        self,
        model_name: str = cfg.CLIP_MODEL_NAME,
        pretrained: str = cfg.CLIP_PRETRAINED,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Loading CLIP model '{model_name}' (pretrained='{pretrained}') on {self.device} …")
 
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = self.model.to(self.device).eval()
        log.info("CLIP model loaded.")
 
    @torch.no_grad()
    def encode_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Return L2-normalised text embeddings, shape (N, D)."""
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Text embeddings"):
            batch = texts[i : i + batch_size]
            tokens = self.tokenizer(batch).to(self.device)
            feats  = self.model.encode_text(tokens)
            feats  = feats / feats.norm(dim=-1, keepdim=True)
            all_embeddings.append(feats.cpu().numpy())
        return np.vstack(all_embeddings).astype(np.float32)
 
    @torch.no_grad()
    def encode_images(self, image_paths: list[str], batch_size: int = 32) -> tuple[np.ndarray, list[int]]:
        """
        Return L2-normalised image embeddings and the indices of successfully
        encoded images (some paths may be missing / corrupt).
        """
        all_embeddings = []
        valid_indices  = []
 
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Image embeddings"):
            batch_paths  = image_paths[i : i + batch_size]
            batch_tensors = []
            batch_valid   = []
 
            for j, p in enumerate(batch_paths):
                try:
                    img = Image.open(p).convert("RGB")
                    batch_tensors.append(self.preprocess(img))
                    batch_valid.append(i + j)
                except Exception:
                    pass  # missing / corrupt image – skip
 
            if not batch_tensors:
                continue
 
            imgs  = torch.stack(batch_tensors).to(self.device)
            feats = self.model.encode_image(imgs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_embeddings.append(feats.cpu().numpy())
            valid_indices.extend(batch_valid)
 
        if all_embeddings:
            return np.vstack(all_embeddings).astype(np.float32), valid_indices
        return np.empty((0, cfg.EMBEDDING_DIM), dtype=np.float32), []
 
    @torch.no_grad()
    def encode_single_image(self, image: Union[str, Path, Image.Image]) -> np.ndarray:
        """Encode a single image (path or PIL). Returns shape (D,)."""
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        feat   = self.model.encode_image(tensor)
        feat   = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).cpu().numpy().astype(np.float32)
 
    @torch.no_grad()
    def encode_single_text(self, text: str) -> np.ndarray:
        """Encode a single text query. Returns shape (D,)."""
        tokens = self.tokenizer([text]).to(self.device)
        feat   = self.model.encode_text(tokens)
        feat   = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).cpu().numpy().astype(np.float32)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 2. ChromaDB store
# ─────────────────────────────────────────────────────────────────────────────
 
class ProductVectorStore:
    """
    Manages two ChromaDB collections:
    - text_collection  : CLIP text embeddings of product descriptions
    - image_collection : CLIP image embeddings of product images
 
    ChromaDB uses cosine similarity by default (HNSW index).
    """
 
    def __init__(self, persist_dir: Path = cfg.CHROMA_DIR):
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.text_col = self.client.get_or_create_collection(
            name=cfg.CHROMA_COLLECTION_TEXT,
            metadata={"hnsw:space": "cosine"},
        )
        self.image_col = self.client.get_or_create_collection(
            name=cfg.CHROMA_COLLECTION_IMAGE,
            metadata={"hnsw:space": "cosine"},
        )
        log.info(
            f"ChromaDB ready | "
            f"text={self.text_col.count()} docs | "
            f"image={self.image_col.count()} docs"
        )
 
    # ── Ingestion ──────────────────────────────────────────────────────────
 
    def add_text_embeddings(
        self,
        product_ids: list[str],
        embeddings: np.ndarray,
        metadatas: list[dict],
    ) -> None:
        """Upsert text embeddings in batches (ChromaDB limit = 5000/call)."""
        batch_size = 1000
        for i in range(0, len(product_ids), batch_size):
            self.text_col.upsert(
                ids=product_ids[i : i + batch_size],
                embeddings=embeddings[i : i + batch_size].tolist(),
                metadatas=metadatas[i : i + batch_size],
            )
        log.info(f"  Stored {len(product_ids)} text embeddings in ChromaDB.")
 
    def add_image_embeddings(
        self,
        product_ids: list[str],
        embeddings: np.ndarray,
        metadatas: list[dict],
    ) -> None:
        batch_size = 1000
        for i in range(0, len(product_ids), batch_size):
            self.image_col.upsert(
                ids=product_ids[i : i + batch_size],
                embeddings=embeddings[i : i + batch_size].tolist(),
                metadatas=metadatas[i : i + batch_size],
            )
        log.info(f"  Stored {len(product_ids)} image embeddings in ChromaDB.")
 
    # ── Retrieval ──────────────────────────────────────────────────────────
 
    def query_text(self, query_embedding: np.ndarray, top_k: int = cfg.TOP_K) -> list[dict]:
        """Search text collection with a query embedding."""
        results = self.text_col.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["metadatas", "distances"],
        )
        return self._format_results(results)
 
    def query_image(self, query_embedding: np.ndarray, top_k: int = cfg.TOP_K) -> list[dict]:
        """Search image collection with a query embedding."""
        results = self.image_col.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self.image_col.count()),
            include=["metadatas", "distances"],
        )
        return self._format_results(results)
 
    def query_hybrid(
        self,
        text_embedding: np.ndarray,
        image_embedding: Optional[np.ndarray] = None,
        top_k: int = cfg.TOP_K,
        text_weight: float = 0.5,
    ) -> list[dict]:
        """
        Hybrid retrieval: merge text and image scores via weighted reciprocal
        rank fusion (RRF).  If no image embedding, falls back to text-only.
        """
        text_results = self.query_text(text_embedding, top_k=top_k * 2)
 
        if image_embedding is None or self.image_col.count() == 0:
            return text_results[:top_k]
 
        image_results = self.query_image(image_embedding, top_k=top_k * 2)
 
        # Reciprocal Rank Fusion
        rrf_scores: dict[str, float] = {}
        k_rrf = 60  # standard RRF constant
 
        for rank, item in enumerate(text_results):
            pid = item["product_id"]
            rrf_scores[pid] = rrf_scores.get(pid, 0) + text_weight / (k_rrf + rank + 1)
 
        image_weight = 1 - text_weight
        for rank, item in enumerate(image_results):
            pid = item["product_id"]
            rrf_scores[pid] = rrf_scores.get(pid, 0) + image_weight / (k_rrf + rank + 1)
 
        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
 
        # Reconstruct result dicts (use text metadata as source of truth)
        id_to_meta = {r["product_id"]: r for r in text_results + image_results}
        return [id_to_meta[pid] for pid in sorted_ids if pid in id_to_meta]
 
    @staticmethod
    def _format_results(chroma_results: dict) -> list[dict]:
        ids        = chroma_results["ids"][0]
        distances  = chroma_results["distances"][0]
        metadatas  = chroma_results["metadatas"][0]
        out = []
        for pid, dist, meta in zip(ids, distances, metadatas):
            out.append({
                "product_id": pid,
                "similarity": round(1 - dist, 4),  # cosine distance → similarity
                **meta,
            })
        return out
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 3. Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────
 
def evaluate_retrieval(
    store: ProductVectorStore,
    encoder: CLIPEncoder,
    df: pd.DataFrame,
    n_queries: int = 200,
    cutoffs: list[int] = cfg.RECALL_CUTOFFS,
) -> pd.DataFrame:
    """
    Evaluate Recall@K using synthetic queries.
 
    Strategy: for each evaluation product, use its combined_description as the
    query and check whether the product itself appears in the top-K results.
    This is the standard self-retrieval evaluation for embedding systems.
 
    Returns a DataFrame with Recall@1, Recall@5, Recall@10.
    """
    log.info(f"Evaluating retrieval on {n_queries} synthetic queries …")
    sample = df.sample(min(n_queries, len(df)), random_state=42)
 
    hits = {k: 0 for k in cutoffs}
 
    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="Eval"):
        query_emb = encoder.encode_single_text(row["combined_description"])
        results   = store.query_text(query_emb, top_k=max(cutoffs))
        true_id = str(row["product_id"])
        result_ids = [str(r["product_id"]) for r in results]
        for k in cutoffs:
            if true_id in result_ids[:k]:
                hits[k] += 1
                
 
    metrics = {f"Recall@{k}": round(hits[k] / len(sample), 4) for k in cutoffs}
    metrics["n_queries"] = len(sample)
    metrics_df = pd.DataFrame([metrics])
 
    log.info("\nRetrieval Evaluation Results:")
    for k in cutoffs:
        log.info(f"  Recall@{k:2d} = {metrics[f'Recall@{k}']:.4f}")
 
    return metrics_df
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 4. Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
 
def build_vector_store(df: pd.DataFrame) -> tuple[ProductVectorStore, CLIPEncoder]:
    """
    Full pipeline: encode all text + images, store in ChromaDB.
 
    Parameters
    ----------
    df : pd.DataFrame
        Processed product dataframe from Task 1.
 
    Returns
    -------
    (store, encoder) tuple for use in Task 3.
    """
    encoder = CLIPEncoder()
    store   = ProductVectorStore()
 
    # ── Text embeddings ────────────────────────────────────────────────────
    if store.text_col.count() < len(df):
        log.info("Generating text embeddings …")
        texts    = df["combined_description"].tolist()
        text_emb = encoder.encode_texts(texts)
 
        metadatas = df.apply(
            lambda r: {
                "product_name": str(r.get("product_name", ""))[:500],
                "brand":        str(r.get("brand", ""))[:200],
                "category":     str(r.get("category", ""))[:200],
                "price_usd":    str(r.get("price_usd", "")),
                "image_path":   str(r.get("local_image_path", "")) or "",
                "description":  str(r.get("combined_description", ""))[:1000],
            },
            axis=1,
        ).tolist()
        store.add_text_embeddings(df["product_id"].astype(str).tolist(), text_emb, metadatas)
    else:
        log.info("Text embeddings already in ChromaDB – skipping.")
 
    # ── Image embeddings ───────────────────────────────────────────────────
    if "local_image_path" in df.columns and store.image_col.count() < df["local_image_path"].notna().sum():
        img_df  = df[df["local_image_path"].notna()].copy()
        log.info(f"Generating image embeddings for {len(img_df)} products …")
 
        img_emb, valid_idx = encoder.encode_images(img_df["local_image_path"].tolist())
 
        valid_rows = img_df.iloc[valid_idx]
        img_meta   = valid_rows.apply(
            lambda r: {
                "product_name": str(r.get("product_name", ""))[:500],
                "brand":        str(r.get("brand", ""))[:200],
                "category":     str(r.get("category", ""))[:200],
                "image_path":   str(r.get("local_image_path", "")),
                "description":  str(r.get("combined_description", ""))[:1000],
            },
            axis=1,
        ).tolist()
        store.add_image_embeddings(valid_rows["product_id"].astype(str).tolist(), img_emb, img_meta)
    else:
        log.info("Image embeddings already in ChromaDB – skipping.")
 
    return store, encoder
 
 
def retrieve(
    query_text: Optional[str] = None,
    query_image: Optional[Union[str, Path, Image.Image]] = None,
    store: Optional[ProductVectorStore] = None,
    encoder: Optional[CLIPEncoder] = None,
    top_k: int = cfg.TOP_K,
) -> list[dict]:
    """
    Convenience wrapper for the retrieval step.
 
    Parameters
    ----------
    query_text  : free-text query string
    query_image : path to image or PIL Image object
    store       : ProductVectorStore (will instantiate if None)
    encoder     : CLIPEncoder (will instantiate if None)
    top_k       : number of results to return
 
    Returns
    -------
    List of dicts with product metadata, sorted by similarity.
    """
    if store is None:
        store = ProductVectorStore()
    if encoder is None:
        encoder = CLIPEncoder()
 
    text_emb  = encoder.encode_single_text(query_text)  if query_text  else None
    image_emb = encoder.encode_single_image(query_image) if query_image else None
 
    if text_emb is not None and image_emb is not None:
        return store.query_hybrid(text_emb, image_emb, top_k=top_k)
    elif image_emb is not None:
        return store.query_image(image_emb, top_k=top_k)
    elif text_emb is not None:
        return store.query_text(text_emb, top_k=top_k)
    else:
        raise ValueError("Provide at least one of query_text or query_image.")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 5. CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    processed_path = cfg.PROCESSED_DIR / "products_processed.csv"
 
    if not processed_path.exists():
        log.error(
            f"Processed CSV not found at {processed_path}. "
            "Run task1_data_preprocessing.py first."
        )
        sys.exit(1)
 
    df = pd.read_csv(processed_path)
    store, encoder = build_vector_store(df)
 
    # Run evaluation
    eval_results = evaluate_retrieval(store, encoder, df)
    eval_path    = cfg.PROCESSED_DIR / "retrieval_evaluation.csv"
    eval_results.to_csv(eval_path, index=False)
    log.info(f"Evaluation results saved to {eval_path}")
 
    # Demo query
    log.info("\nDemo query: 'wireless noise cancelling headphones'")
    results = retrieve("wireless noise cancelling headphones", store=store, encoder=encoder)
    for i, r in enumerate(results, 1):
        log.info(f"  {i}. [{r['similarity']:.3f}] {r.get('product_name', '')[:80]}")
 