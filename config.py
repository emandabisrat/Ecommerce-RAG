import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
RAW_DATA_DIR    = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"
CHROMA_DIR      = BASE_DIR / "chroma_db"
IMAGE_CACHE_DIR = DATA_DIR / "images"

for d in [RAW_DATA_DIR, PROCESSED_DIR, CHROMA_DIR, IMAGE_CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DATASET_NAME         = "promptcloud/amazon-product-dataset-2020"
MAX_PRODUCTS         = 5000      
MIN_DESCRIPTION_LEN  = 20       

TEXT_ATTRIBUTES = [
    "product_name",    
    "brand",           
    "category",       
    "selling_price",  
    "about_product",  
    "product_specification", 
]

CLIP_MODEL_NAME = "ViT-B-32"          
CLIP_PRETRAINED = "openai"           
EMBEDDING_DIM   = 512                

CHROMA_COLLECTION_TEXT  = "product_text_embeddings"
CHROMA_COLLECTION_IMAGE = "product_image_embeddings"

TOP_K           = 5      
RETRIEVAL_MODE  = "text" 

OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")
LLM_MODEL      = "llava"
LLM_MAX_TOKENS = 1024

RECALL_CUTOFFS = [1, 5, 10]