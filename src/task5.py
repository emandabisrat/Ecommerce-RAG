"""
Task 5 – Evaluation: Retrieval Accuracy + Response Relevance
=============================================================
Metrics
-------
1. Recall@1, Recall@5, Recall@10   – retrieval accuracy (from Task 2)
2. Mean Reciprocal Rank (MRR)       – how highly the correct product is ranked
3. Response Relevance Score         – does the LLM answer actually address the question?
4. Hallucination Rate               – does the LLM say things not in the retrieved context?
5. Image Retrieval Accuracy         – does the correct image get returned on image requests?

Run with:
    python -m src.task5_evaluation
"""

import logging
import sys
import re
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import ollama
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from src.task2_embeddings_and_rag import (
    CLIPEncoder,
    ProductVectorStore,
    build_vector_store,
    evaluate_retrieval,
    retrieve,
)
from src.task3_llm_integration import (
    MultimodalChatbot,
    _format_product_context,
    few_shot_prompt,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Mean Reciprocal Rank (MRR)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_mrr(
    store: ProductVectorStore,
    encoder: CLIPEncoder,
    df: pd.DataFrame,
    n_queries: int = 200,
    top_k: int = 10,
) -> float:
    """
    MRR measures the rank of the first correct result.
    MRR = 1/N * sum(1 / rank_of_correct_item)
    Score of 1.0 = always retrieved first. 0.0 = never retrieved.
    """
    log.info(f"Computing MRR on {n_queries} queries …")
    sample = df.sample(min(n_queries, len(df)), random_state=42)
    reciprocal_ranks = []

    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="MRR"):
        query_emb  = encoder.encode_single_text(row["combined_description"])
        results    = store.query_text(query_emb, top_k=top_k)
        result_ids = [r["product_id"] for r in results]

        try:
            rank = result_ids.index(str(row["product_id"])) + 1
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            reciprocal_ranks.append(0.0)  # not found in top_k

    mrr = float(np.mean(reciprocal_ranks))
    log.info(f"  MRR@{top_k} = {mrr:.4f}")
    return mrr


# ─────────────────────────────────────────────────────────────────────────────
# 2. Response Relevance + Hallucination (LLM-as-judge)
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are an evaluator for a product assistant chatbot.
Given a user question, retrieved product context, and the assistant's response,
score the response on two dimensions.

Return ONLY valid JSON with this exact structure, nothing else:
{{
  "relevance_score": <float 0.0-1.0>,
  "hallucination_score": <float 0.0-1.0>,
  "relevance_reason": "<one sentence>",
  "hallucination_reason": "<one sentence>"
}}

Scoring guide:
- relevance_score: 1.0 = response directly and fully answers the question using the context.
                   0.5 = partially answers or is vague.
                   0.0 = off-topic or doesn't answer.
- hallucination_score: 0.0 = response only uses facts from the context (good).
                        0.5 = minor extrapolation beyond context.
                        1.0 = response invents facts not in the context (bad).

User Question: {question}

Retrieved Context:
{context}

Assistant Response:
{response}"""


TEST_QUERIES = [
    "What are the features of this product?",
    "Is this product waterproof?",
    "What is the battery life?",
    "Can you compare these two products?",
    "What colors does this come in?",
    "What is the price?",
    "Is this compatible with Alexa?",
    "What is the warranty?",
    "What is the weight of this product?",
    "Does this product come with accessories?",
]


def evaluate_response_quality(
    store: ProductVectorStore,
    encoder: CLIPEncoder,
    df: pd.DataFrame,
    n_queries: int = 10,
    top_k: int = cfg.TOP_K,
) -> pd.DataFrame:
    """
    Use LLaVA itself as a judge to score response relevance and hallucination.
    Samples n_queries products, generates a response, then asks LLaVA to score it.
    """
    log.info(f"Evaluating response quality on {n_queries} queries …")
    sample = df.sample(min(n_queries, len(df)), random_state=99)
    records = []

    for i, (_, row) in enumerate(tqdm(sample.iterrows(), total=len(sample), desc="Response eval")):
        # Pick a test question
        question = TEST_QUERIES[i % len(TEST_QUERIES)]
        product_name = row.get("product_name", "this product")
        full_question = question.replace("this product", product_name).replace("these two products", product_name)

        # Retrieve context
        query_emb = encoder.encode_single_text(full_question)
        results   = store.query_text(query_emb, top_k=top_k)
        context   = _format_product_context(results)

        # Generate response
        prompt = few_shot_prompt(full_question, context)
        try:
            resp = ollama.chat(
                model=cfg.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = resp["message"]["content"]
        except Exception as e:
            log.warning(f"LLM call failed: {e}")
            response_text = ""

        # Judge the response
        judge_input = JUDGE_PROMPT.format(
            question=full_question,
            context=context[:1500],
            response=response_text[:800],
        )
        try:
            judge_resp = ollama.chat(
                model=cfg.LLM_MODEL,
                messages=[{"role": "user", "content": judge_input}],
            )
            raw = judge_resp["message"]["content"].strip()
            # Strip markdown fences if present
            raw = re.sub(r"```json|```", "", raw).strip()
            scores = json.loads(raw)
        except Exception as e:
            log.warning(f"Judge parsing failed: {e}")
            scores = {
                "relevance_score": None,
                "hallucination_score": None,
                "relevance_reason": "parse error",
                "hallucination_reason": "parse error",
            }

        records.append({
            "product_name":        product_name,
            "question":            full_question,
            "response":            response_text[:300] + "…" if len(response_text) > 300 else response_text,
            "relevance_score":     scores.get("relevance_score"),
            "hallucination_score": scores.get("hallucination_score"),
            "relevance_reason":    scores.get("relevance_reason"),
            "hallucination_reason": scores.get("hallucination_reason"),
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Image Retrieval Accuracy
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_image_retrieval(
    store: ProductVectorStore,
    encoder: CLIPEncoder,
    df: pd.DataFrame,
    n_queries: int = 100,
) -> dict:
    """
    For products that have images, check if querying by product name
    returns the correct product's image in top-1 and top-5.
    """
    log.info(f"Evaluating image retrieval on {n_queries} queries …")
    img_df = df[df["local_image_path"].notna()].copy()
    sample = img_df.sample(min(n_queries, len(img_df)), random_state=42)

    hits_1, hits_5 = 0, 0

    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="Image retrieval"):
        query = str(row.get("product_name", ""))
        if not query:
            continue
        query_emb  = encoder.encode_single_text(query)
        results    = store.query_image(query_emb, top_k=5)
        result_ids = [r["product_id"] for r in results]

        if str(row["product_id"]) in result_ids[:1]:
            hits_1 += 1
        if str(row["product_id"]) in result_ids[:5]:
            hits_5 += 1

    total = len(sample)
    metrics = {
        "Image Recall@1": round(hits_1 / total, 4),
        "Image Recall@5": round(hits_5 / total, 4),
        "n_queries":      total,
    }
    log.info(f"  Image Recall@1 = {metrics['Image Recall@1']:.4f}")
    log.info(f"  Image Recall@5 = {metrics['Image Recall@5']:.4f}")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 4. Summary report
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(
    recall_df:    pd.DataFrame,
    mrr:          float,
    response_df:  pd.DataFrame,
    image_metrics: dict,
) -> None:
    print("\n" + "=" * 60)
    print("           EVALUATION SUMMARY REPORT")
    print("=" * 60)

    print("\n── Text Retrieval ──────────────────────────────────────────")
    for col in recall_df.columns:
        if col != "n_queries":
            print(f"  {col:12s} = {recall_df[col].iloc[0]:.4f}")
    print(f"  MRR@10       = {mrr:.4f}")
    print(f"  (n={int(recall_df['n_queries'].iloc[0])} queries)")

    print("\n── Image Retrieval ─────────────────────────────────────────")
    print(f"  Recall@1     = {image_metrics['Image Recall@1']:.4f}")
    print(f"  Recall@5     = {image_metrics['Image Recall@5']:.4f}")
    print(f"  (n={image_metrics['n_queries']} queries)")

    print("\n── Response Quality ────────────────────────────────────────")
    valid = response_df.dropna(subset=["relevance_score", "hallucination_score"])
    if len(valid):
        avg_rel   = valid["relevance_score"].mean()
        avg_hall  = valid["hallucination_score"].mean()
        print(f"  Avg Relevance Score     = {avg_rel:.4f}  (1.0 = best)")
        print(f"  Avg Hallucination Score = {avg_hall:.4f}  (0.0 = best)")
        print(f"  (n={len(valid)} responses evaluated)")
    else:
        print("  No valid response scores (judge parsing failed)")

    print("\n── Interpretation ──────────────────────────────────────────")
    recall1 = recall_df["Recall@1"].iloc[0]
    if recall1 >= 0.8:
        print("  ✅ Retrieval is strong — correct product found first >80% of time")
    elif recall1 >= 0.5:
        print("  ⚠️  Retrieval is moderate — consider re-tuning embeddings")
    else:
        print("  ❌ Retrieval needs improvement — check combined_description quality")

    if len(valid):
        if avg_rel >= 0.7:
            print("  ✅ Responses are relevant and grounded in context")
        elif avg_rel >= 0.4:
            print("  ⚠️  Responses partially answer questions — try multi_shot strategy")
        else:
            print("  ❌ Responses are off-topic — review prompt templates")

        if avg_hall <= 0.2:
            print("  ✅ Low hallucination — model stays within retrieved context")
        elif avg_hall <= 0.5:
            print("  ⚠️  Some hallucination detected — tighten system prompt")
        else:
            print("  ❌ High hallucination — model is inventing facts")

    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    processed_path = cfg.PROCESSED_DIR / "products_processed.csv"
    if not processed_path.exists():
        log.error(f"No processed CSV at {processed_path}. Run task1 first.")
        sys.exit(1)

    df = pd.read_csv(processed_path)
    store, encoder = build_vector_store(df)

    # 1. Recall@K (already in task2, reuse)
    log.info("\n[1/4] Running Recall@K evaluation …")
    recall_df = evaluate_retrieval(store, encoder, df, n_queries=200)

    # 2. MRR
    log.info("\n[2/4] Running MRR evaluation …")
    mrr = evaluate_mrr(store, encoder, df, n_queries=200)

    # 3. Response relevance + hallucination
    log.info("\n[3/4] Running response quality evaluation …")
    response_df = evaluate_response_quality(store, encoder, df, n_queries=10)

    # 4. Image retrieval
    log.info("\n[4/4] Running image retrieval evaluation …")
    image_metrics = evaluate_image_retrieval(store, encoder, df, n_queries=100)

    # Print summary
    print_summary(recall_df, mrr, response_df, image_metrics)

    # Save results
    out_dir = cfg.PROCESSED_DIR
    recall_df.to_csv(out_dir / "eval_retrieval.csv", index=False)
    response_df.to_csv(out_dir / "eval_response_quality.csv", index=False)
    pd.DataFrame([{"MRR@10": mrr, **image_metrics}]).to_csv(
        out_dir / "eval_summary.csv", index=False
    )
    log.info(f"Results saved to {out_dir}")