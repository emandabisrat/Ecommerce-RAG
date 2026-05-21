"""
Task 1 – Understanding and Preprocessing Multimodal Data
=========================================================
Objectives
----------
1. Load the Amazon Product Dataset 2020 from Kaggle.
2. Analyse and choose the optimal combination of product attributes.
3. Clean, normalise, and build a rich combined text description per product.
4. Download and cache product images where available.
5. Export a clean processed CSV ready for embedding.
 
Run standalone:
    python -m src.task1_data_preprocessing
"""
 
import re
import os
import io
import logging
import hashlib
from pathlib import Path
from typing import Optional
 
import pandas as pd
import numpy as np
import requests
from PIL import Image
from tqdm import tqdm
 
# Add parent to path when running standalone
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
 
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 1. Load dataset
# ─────────────────────────────────────────────────────────────────────────────
 
def load_raw_dataset(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load dataset from a local CSV or attempt Kaggle download.
 
    The Amazon Product Dataset 2020 CSV columns vary by version, but
    typically include: uniq_id, product_name, brand, category,
    selling_price, about_product, product_specification, image (URL),
    product_url, rating, number_of_reviews.
    """
    if csv_path and Path(csv_path).exists():
        log.info(f"Loading dataset from {csv_path}")
        df = pd.read_csv(csv_path, low_memory=False)
    else:
        # Try Kaggle API
        try:
            import kaggle
            log.info("Downloading dataset from Kaggle …")
            kaggle.api.dataset_download_files(
                cfg.DATASET_NAME,
                path=str(cfg.RAW_DATA_DIR),
                unzip=True,
            )
            csv_files = list(cfg.RAW_DATA_DIR.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError("No CSV found after Kaggle download.")
            df = pd.read_csv(csv_files[0], low_memory=False)
        except Exception as e:
            log.error(
                f"Could not load dataset automatically: {e}\n"
                "Please download manually from:\n"
                "  https://www.kaggle.com/datasets/promptcloud/amazon-product-dataset-2020\n"
                f"and place the CSV in {cfg.RAW_DATA_DIR}"
            )
            raise
 
    log.info(f"Raw dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 2. Attribute analysis helper
# ─────────────────────────────────────────────────────────────────────────────
 
def analyse_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Print a quick completeness/uniqueness report for every column.
    Helps decide which attributes to include in the product description.
    """
    report_rows = []
    for col in df.columns:
        non_null = df[col].notna().sum()
        unique   = df[col].nunique()
        pct_fill = 100 * non_null / len(df)
        avg_len  = (
            df[col].dropna().astype(str).str.len().mean()
            if df[col].dtype == object else None
        )
        report_rows.append(
            {
                "column": col,
                "filled_%": round(pct_fill, 1),
                "unique_values": unique,
                "avg_text_length": round(avg_len, 1) if avg_len else None,
            }
        )
    report = pd.DataFrame(report_rows).sort_values("filled_%", ascending=False)
    log.info("\nAttribute Analysis:\n" + report.to_string(index=False))
    return report
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 3. Cleaning helpers
# ─────────────────────────────────────────────────────────────────────────────
 
def _clean_text(text) -> str:
    """Basic text normalisation."""
    if pd.isna(text):
        return ""
    text = str(text).strip()
    # collapse whitespace
    text = re.sub(r"\s+", " ", text)
    # remove non-printable characters
    text = re.sub(r"[^\x20-\x7E]", "", text)
    return text
 
 
def _parse_price(price_str) -> Optional[float]:
    """Extract numeric value from price strings like '$29.99' or '29.99'."""
    if pd.isna(price_str):
        return None
    match = re.search(r"[\d,]+\.?\d*", str(price_str).replace(",", ""))
    return float(match.group()) if match else None
 
 
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalise the raw dataframe.
 
    Steps:
    - Standardise column names to snake_case.
    - Drop duplicate products.
    - Parse price to numeric.
    - Strip whitespace / bad characters from text columns.
    - Drop rows with missing product name.
    - Optionally limit to MAX_PRODUCTS.
    """
    log.info("Cleaning dataframe …")
 
    # Normalise column names
    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
    )
 
    # Drop exact duplicates
    before = len(df)
    df = df.drop_duplicates()
    log.info(f"  Dropped {before - len(df)} exact duplicate rows")
 
    # Parse price
    if "selling_price" in df.columns:
        df["price_usd"] = df["selling_price"].apply(_parse_price)
 
    # Clean text columns
    text_cols = [c for c in cfg.TEXT_ATTRIBUTES if c in df.columns]
    for col in text_cols:
        df[col] = df[col].apply(_clean_text)
 
    # Drop products with no name
    if "product_name" in df.columns:
        df = df[df["product_name"].str.len() >= 3]
 
    # Drop products with very short combined description
    combined_length = df[[c for c in text_cols if c in df.columns]].apply(
        lambda row: " ".join(row.values.astype(str)), axis=1
    ).str.len()
    df = df[combined_length >= cfg.MIN_DESCRIPTION_LEN]
 
    # Limit size for prototyping
    if cfg.MAX_PRODUCTS:
        df = df.head(cfg.MAX_PRODUCTS)
 
    # Reset index and create a stable product_id
    df = df.reset_index(drop=True)
    df["product_id"] = df.index.astype(str).str.zfill(6)
 
    log.info(f"  Final shape after cleaning: {df.shape}")
    return df
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 4. Build rich combined text description
# ─────────────────────────────────────────────────────────────────────────────
 
def build_product_description(row: pd.Series) -> str:
    """
    Combine selected product attributes into a single natural-language
    text description optimised for CLIP / LLM embedding.
 
    Attribute selection rationale
    ------------------------------
    • product_name  – primary identifier; always included
    • brand         – narrows semantic search (e.g. 'Apple' vs generic)
    • category      – coarse-grained topic signal
    • selling_price – useful for price-range queries
    • about_product – rich feature bullet points; highest information density
    • product_specification – detailed specs; crucial for comparison queries
 
    Attributes intentionally omitted from description:
    • product_url   – not semantic
    • image         – handled separately as an image embedding
    • rating / reviews – volatile; not product identity
    """
    parts = []
 
    if pd.notna(row.get("product_name")) and row["product_name"]:
        parts.append(f"Product: {row['product_name']}")
 
    if pd.notna(row.get("brand")) and row["brand"] not in ("", "nan"):
        parts.append(f"Brand: {row['brand']}")
 
    if pd.notna(row.get("category")) and row["category"] not in ("", "nan"):
        parts.append(f"Category: {row['category']}")
 
    if pd.notna(row.get("price_usd")):
        parts.append(f"Price: ${row['price_usd']:.2f}")
    elif pd.notna(row.get("selling_price")) and row["selling_price"]:
        parts.append(f"Price: {row['selling_price']}")
 
    if pd.notna(row.get("about_product")) and row["about_product"] not in ("", "nan"):
        parts.append(f"Features: {row['about_product']}")
 
    if pd.notna(row.get("product_specification")) and row["product_specification"] not in ("", "nan"):
        # Truncate very long specs to keep embeddings focused
        spec = row["product_specification"][:800]
        parts.append(f"Specifications: {spec}")
 
    return " | ".join(parts)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 5. Image downloading
# ─────────────────────────────────────────────────────────────────────────────
 
def _image_filename(product_id: str, url: str) -> Path:
    ext = Path(url.split("?")[0]).suffix or ".jpg"
    return cfg.IMAGE_CACHE_DIR / f"{product_id}{ext}"
 
 
def download_image(url: str, product_id: str, timeout: int = 8) -> Optional[Path]:
    """Download a product image and cache it locally. Returns local path or None."""
    dest = _image_filename(product_id, url)
    if dest.exists():
        return dest
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        img.save(dest)
        return dest
    except Exception:
        return None
 
 
def download_images_batch(df: pd.DataFrame, max_images: int = 1000) -> pd.DataFrame:
    """
    Download images for up to `max_images` products.
    Adds a column `local_image_path` to the dataframe.
    """
    image_col = next((c for c in ["image", "image_url", "images"] if c in df.columns), None)
    if image_col is None:
        log.warning("No image URL column found – skipping image download.")
        df["local_image_path"] = None
        return df
 
    log.info(f"Downloading up to {max_images} product images …")
    paths = []
    subset = df.head(max_images)
 
    for _, row in tqdm(subset.iterrows(), total=len(subset), desc="Images"):
        url_raw = str(row.get(image_col, ""))
        # some rows have multiple URLs separated by '|'
        url = url_raw.split("|")[0].strip()
        if url.startswith("http"):
            local = download_image(url, row["product_id"])
            paths.append(str(local) if local else None)
        else:
            paths.append(None)
 
    # Pad remainder with None
    paths.extend([None] * (len(df) - len(paths)))
    df["local_image_path"] = paths
 
    n_downloaded = sum(1 for p in paths if p)
    log.info(f"  Successfully cached {n_downloaded}/{min(max_images, len(df))} images")
    return df
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 6. Quality checks
# ─────────────────────────────────────────────────────────────────────────────
 
def quality_report(df: pd.DataFrame) -> None:
    """Log a summary of the processed dataset."""
    log.info("\n── Processed Dataset Quality Report ─────────────────────────────")
    log.info(f"  Total products      : {len(df)}")
    log.info(f"  Unique categories   : {df['category'].nunique() if 'category' in df.columns else 'N/A'}")
    log.info(f"  Products with image : {df['local_image_path'].notna().sum() if 'local_image_path' in df.columns else 'N/A'}")
    log.info(f"  Avg description len : {df['combined_description'].str.len().mean():.0f} chars")
    log.info(f"  Price range         : ${df['price_usd'].min():.2f} – ${df['price_usd'].max():.2f}" if "price_usd" in df.columns else "")
    log.info("─────────────────────────────────────────────────────────────────\n")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 7. Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
 
def run_preprocessing(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    End-to-end preprocessing pipeline.
 
    Parameters
    ----------
    csv_path : str, optional
        Path to the raw CSV file. If None, will try Kaggle download.
 
    Returns
    -------
    pd.DataFrame
        Processed dataframe saved to `cfg.PROCESSED_DIR / 'products_processed.csv'`
    """
    # Step 1 – Load
    df = load_raw_dataset(Path(csv_path) if csv_path else None)
 
    # Step 2 – Analyse (informational)
    analyse_attributes(df)
 
    # Step 3 – Clean
    df = clean_dataframe(df)
 
    # Step 4 – Build combined text description
    log.info("Building combined product descriptions …")
    df["combined_description"] = df.apply(build_product_description, axis=1)
 
    # Step 5 – Download images (limit to 500 for prototyping)
    df = download_images_batch(df, max_images=500)
 
    # Step 6 – Quality check
    quality_report(df)
 
    # Step 7 – Save
    out_path = cfg.PROCESSED_DIR / "products_processed.csv"
    df.to_csv(out_path, index=False)
    log.info(f"Saved processed dataset to {out_path}")
 
    return df
 
 
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Task 1 – Data Preprocessing")
    parser.add_argument("--csv", type=str, default=None, help="Path to raw CSV file")
    args = parser.parse_args()
    run_preprocessing(csv_path=args.csv)