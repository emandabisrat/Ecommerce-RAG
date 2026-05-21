# Ecommerce Multi-Modal Product Chatbot 
<img width="900" height="500" alt="image" src="https://github.com/user-attachments/assets/fd2d87ec-3c75-472c-8040-7f73bcd30a0f" />

Amazon Product Dataset 2020 · CLIP · ChromaDB · Claude
 
A multimodal RAG chatbot that answers product questions using both text and
image inputs, powered by CLIP embeddings and Claude as the LLM backbone.
 
---
 
## Project Structure
 
```
multimodal-chatbot/
├── data/
│   ├── raw/              ← place downloaded Kaggle CSV here
│   ├── processed/        ← auto-generated cleaned CSV + eval results
│   └── images/           ← auto-downloaded product images
├── chroma_db/            ← persistent ChromaDB vector store
├── src/
│   ├── task1_data_preprocessing.py
│   ├── task2_embeddings_and_rag.py
│   └── task3_llm_integration.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_rag_pipeline.ipynb
├── config.py
├── requirements.txt
├── .env.example
└── README.md
```
 
---
 
## Setup
 
### 1. Clone & install dependencies
 
```bash
git clone <your-repo-url>
cd multimodal-chatbot
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
 
### 2. Configure API keys
 
```bash
cp .env.example .env
# Edit .env and fill in:
#   ANTHROPIC_API_KEY  → https://console.anthropic.com/
#   KAGGLE_USERNAME    → from https://www.kaggle.com/account
#   KAGGLE_KEY         → from https://www.kaggle.com/account
```
 
### 3. Download the dataset
 
**Option A – Automatic (Kaggle API)**
```bash
# Kaggle credentials must be set in .env
python -c "
import kaggle, config as cfg
kaggle.api.dataset_download_files(
    'promptcloud/amazon-product-dataset-2020',
    path=str(cfg.RAW_DATA_DIR), unzip=True
)"
```
 
**Option B – Manual**
1. Go to https://www.kaggle.com/datasets/promptcloud/amazon-product-dataset-2020
2. Download and unzip the CSV
3. Place it in `data/raw/`
---
 
## Running the Pipeline
 
### Task 1 – Data Preprocessing
 
```bash
# Auto-detect CSV in data/raw/
python -m src.task1_data_preprocessing
 
# Or specify path explicitly
python -m src.task1_data_preprocessing --csv data/raw/marketing_sample_for_amazon.csv
```
 
Outputs: `data/processed/products_processed.csv` + cached images in `data/images/`
 
### Task 2 – Build Vector Store & Evaluate
 
```bash
python -m src.task2_embeddings_and_rag
```
 
Outputs:
- ChromaDB collections in `chroma_db/`
- `data/processed/retrieval_evaluation.csv` with Recall@1/5/10
### Task 3 – LLM Chatbot (REPL)
 
```bash
python -m src.task3_llm_integration
```
 
**Query with an image:**
```
You: What is this product? [image:/path/to/product.jpg]
```
 
---
 
## Architecture
 
```
User Query (text + optional image)
        │
        ▼
  ┌─────────────┐
  │  CLIP Encoder│  (openai/clip-vit-base-patch32)
  └──────┬──────┘
         │ 512-dim embeddings
         ▼
  ┌─────────────────────────┐
  │     ChromaDB (HNSW)     │
  │  ┌──────────┐ ┌───────┐ │
  │  │ Text col │ │ Img   │ │
  │  │ (cosine) │ │  col  │ │
  │  └──────────┘ └───────┘ │
  └───────────┬─────────────┘
              │ Top-K products
              ▼
  ┌─────────────────────┐
  │  Prompt Builder     │  (zero-shot / few-shot / multi-shot)
  └──────────┬──────────┘
             │
             ▼
  ┌──────────────────────┐
  │  Claude claude-opus-4-5 (LLM) │
  └──────────┬───────────┘
             │
             ▼
       Grounded Answer
```
 
### Retrieval Modes
 
| Mode    | Description |
|---------|-------------|
| Text    | CLIP text embedding → text collection search |
| Image   | CLIP image embedding → image collection search |
| Hybrid  | Both, merged via Reciprocal Rank Fusion (RRF) |
 
### Prompt Strategies
 
| Strategy   | When to use |
|------------|-------------|
| zero_shot  | Simple factual queries |
| few_shot   | Standard Q&A (default) |
| multi_shot | Comparisons, recommendations |
 
---
 
## Configuration
 
Edit `config.py` to tune:
 
| Parameter       | Default | Description |
|-----------------|---------|-------------|
| `MAX_PRODUCTS`  | 5000    | How many rows to use |
| `TOP_K`         | 5       | Retrieved products per query |
| `CLIP_MODEL_NAME` | ViT-B-32 | CLIP variant |
| `LLM_MODEL`     | claude-opus-4-5 | Claude model |
 
---
 
## Evaluation
 
The retrieval system is evaluated using self-retrieval (each product's description
is used as a query; we check if the product appears in top-K results).
 
Example results:
 
| Metric    | Score |
|-----------|-------|
| Recall@1  | ~0.72 |
| Recall@5  | ~0.91 |
| Recall@10 | ~0.95 |
 
---
 
## References
 
- [CLIP Paper](https://arxiv.org/abs/2103.00020) – Radford et al., 2021
- [RAG Paper](https://arxiv.org/abs/2005.11401) – Lewis et al., 2020
- [open_clip](https://github.com/mlfoundations/open_clip)
- [ChromaDB](https://www.trychroma.com/)
- [Anthropic Claude](https://www.anthropic.com/)
