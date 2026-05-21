"""
Task 3 – Conversational LLM Integration with Google Gemini
===========================================================
Objectives
----------
1. Take a user query (text and/or image) through the RAG pipeline.
2. Format retrieved product context into prompt templates
   (zero-shot, few-shot, multi-shot).
3. Call Gemini 1.5 Flash to generate a grounded, context-aware response.
4. Maintain multi-turn conversation history via Gemini's chat session.
5. Support three query modes: text-only, image-only, hybrid.

Run standalone (REPL demo):
    python -m src.task3_llm_integration
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union

import ollama
import pandas as pd
from PIL import Image
from src.task_image import ProductImageGenerator

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from src.task2_embeddings_and_rag import (
    CLIPEncoder,
    ProductVectorStore,
    retrieve,
    build_vector_store,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a helpful Amazon product assistant.
You answer questions about products using ONLY the retrieved product context provided to you.
If the answer cannot be found in the context, say so clearly — do not invent information.
When asked to compare products, structure your answer clearly.
When asked to recommend, explain your reasoning based on the retrieved data.
If the user provides an image, identify the product and answer based on both the image
and the retrieved context."""

def _format_product_context(retrieved_products: list[dict]) -> str:
    if not retrieved_products:
        return "No relevant products found in the database."

    lines = ["=== Retrieved Product Context ===\n"]
    for i, p in enumerate(retrieved_products, 1):
        lines.append(f"[Product {i}] (similarity: {p.get('similarity', 'N/A')})")
        lines.append(f"  Name     : {p.get('product_name', 'N/A')}")
        lines.append(f"  Brand    : {p.get('brand', 'N/A')}")
        lines.append(f"  Category : {p.get('category', 'N/A')}")
        lines.append(f"  Price    : ${p.get('price_usd', 'N/A')}")
        lines.append(f"  Image    : {p.get('local_image_path', 'N/A')}")  # ← ADD THIS
        desc = p.get("description", "")
        if len(desc) > 600:
            desc = desc[:600] + "…"
        lines.append(f"  Details  : {desc}")
        lines.append("")
    return "\n".join(lines)

import re

IMAGE_INTENT_PATTERNS = [
    r"show me",
    r"can you show",
    r"picture of",
    r"image of",
    r"photo of",
    r"what does .+ look like",
    r"display .+ image",
]

def _is_image_request(text: str) -> bool:
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in IMAGE_INTENT_PATTERNS)

IMAGE_GENERATION_PATTERNS = [
    r"generate",
    r"create",
    r"design",
    r"make an image",
    r"draw",
    r"make a picture",
]
def _is_image_generation_request(text: str) -> bool:
    text_lower = text.lower()

    return any(
        re.search(p, text_lower)
        for p in IMAGE_GENERATION_PATTERNS
    )

def zero_shot_prompt(user_query: str, context: str) -> str:
    return f"""{context}

User Question: {user_query}

Answer the user's question using only the product information provided above."""


FEW_SHOT_EXAMPLES = """
=== Examples ===

Example 1:
User: "What are the features of the Echo Dot?"
Context includes: [Product 1] Name: Amazon Echo Dot (3rd Gen), Details: Features Alexa,
  1.6-inch speaker, Bluetooth, Wi-Fi, compact design.
Answer: The Amazon Echo Dot (3rd Gen) features Alexa voice assistant built-in,
a 1.6-inch speaker for rich sound, Bluetooth for speaker pairing, and Wi-Fi connectivity.
Its compact, fabric design makes it easy to fit anywhere in your home.

Example 2:
User: "Is this waterproof?"
Context includes: [Product 1] Name: Fitbit Charge 4, Details: Water resistant up to 50 metres.
Answer: Yes! The Fitbit Charge 4 is water resistant up to 50 metres, making it suitable
for swimming and showering.

=== End of Examples ===
"""


def few_shot_prompt(user_query: str, context: str) -> str:
    return f"""{FEW_SHOT_EXAMPLES}

{context}

User Question: {user_query}

Following the style of the examples above, answer the user's question using only
the product context provided."""


def multi_shot_prompt(user_query: str, context: str) -> str:
    return f"""{FEW_SHOT_EXAMPLES}

{context}

User Question: {user_query}

Think step-by-step:
1. Identify which retrieved products are relevant to the question.
2. Extract the key facts that answer the question.
3. Formulate a clear, concise response grounded in those facts.

Your answer:"""


PROMPT_STRATEGIES = {
    "zero_shot":  zero_shot_prompt,
    "few_shot":   few_shot_prompt,
    "multi_shot": multi_shot_prompt,
}


def _load_image(image_input: Union[str, Path, Image.Image]) -> Image.Image:
    if isinstance(image_input, Image.Image):
        return image_input
    return Image.open(image_input).convert("RGB")


class MultimodalChatbot:
    """
    End-to-end multimodal product chatbot powered by Google Gemini.

    Workflow for each turn:
    1. Encode the query (text / image / both) with CLIP.
    2. Retrieve top-K relevant products from ChromaDB.
    3. Build a grounded prompt using the chosen strategy.
    4. Send the prompt (+ raw PIL image if provided) to Gemini 1.5 Flash.
    5. Return the response; Gemini's chat session tracks history automatically.
    """
    def __init__(self, store, encoder, df=None, prompt_strategy="few_shot", top_k=cfg.TOP_K):
        self.model = cfg.LLM_MODEL
        self.chat_history = []
        self.store = store
        self.encoder = encoder
        self.top_k = top_k
        self.df = df  # ← ADD THIS
        self.image_generator = ProductImageGenerator()

        if prompt_strategy not in PROMPT_STRATEGIES:
            raise ValueError(f"prompt_strategy must be one of {list(PROMPT_STRATEGIES.keys())}")
        self.prompt_fn = PROMPT_STRATEGIES[prompt_strategy]
        self.prompt_strategy = prompt_strategy
        log.info(f"Chatbot ready | strategy={prompt_strategy} | top_k={top_k} | model={self.model}")
    

    def chat(self, user_text=None, user_image=None, prompt_strategy=None) -> tuple[str, Optional[str]]:
        if not user_text and user_image is None:
            return "Please provide a text query or an image.", None
        if user_text and _is_image_generation_request(user_text):
            generation_prompt = user_text
            generated_image = self.image_generator.generate(generation_prompt)
            return ("Here is the generated product image.", generated_image)

        prompt_fn = PROMPT_STRATEGIES.get(prompt_strategy or "", self.prompt_fn)

        pil_image = _load_image(user_image) if user_image else None
        results = retrieve(
            query_text=user_text,
            query_image=pil_image,
            store=self.store,
            encoder=self.encoder,
            top_k=self.top_k,
        )
        context = _format_product_context(results)
        grounded_query = user_text or "Please identify and describe this product."
        full_prompt = prompt_fn(grounded_query, context)

        message = {"role": "user", "content": full_prompt}
        if user_image:
            import base64, io
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG")
            message["images"] = [base64.b64encode(buffer.getvalue()).decode()]

        self.chat_history.append(message)

        response = ollama.chat(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + self.chat_history,
        )
        assistant_text = response["message"]["content"]
        self.chat_history.append({"role": "assistant", "content": assistant_text})

        # ── Image lookup ───────────────────────────────────────────────────────
        returned_image_path = None
        if user_text and _is_image_request(user_text) and results:
            top_product = results[0]
            image_path = top_product.get("local_image_path")
            if image_path and Path(image_path).exists():
                returned_image_path = image_path
            else:
                assistant_text += "\n\n_(No image available for this product.)_"

        return assistant_text, returned_image_path

    
    def reset(self):
        self.chat_history = []
        log.info("Conversation history cleared.")

    def get_history(self):
        return self.chat_history


def create_chatbot(df=None, prompt_strategy="few_shot") -> MultimodalChatbot:
    if df is None:
        processed_path = cfg.PROCESSED_DIR / "products_processed.csv"
        if not processed_path.exists():
            raise FileNotFoundError(
                f"No processed data at {processed_path}. "
                "Run task1_data_preprocessing.py first."
            )
        df = pd.read_csv(processed_path)

    store, encoder = build_vector_store(df)
    return MultimodalChatbot(store, encoder, df=df, prompt_strategy=prompt_strategy)  # ← pass df

if __name__ == "__main__":
    print("\n🛍️  Multimodal Product Chatbot — REPL Demo (Gemini)")
    print("  Commands: 'reset' | 'quit' | type your question")
    print("  To include an image: your question [image:/path/to/file.jpg]\n")

    chatbot = create_chatbot(prompt_strategy="few_shot")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "reset":
            chatbot.reset()
            print("Conversation reset.\n")
            continue

        image_path = None
        text_part  = user_input
        import re
        match = re.search(r"\[image:(.+?)\]", user_input)
        if match:
            image_path = match.group(1).strip()
            text_part  = user_input.replace(match.group(0), "").strip()

        try:
            response = chatbot.chat(
                user_text=text_part or None,
                user_image=image_path,
            )
            print(f"\nAssistant: {response}\n")
        except Exception as e:
            print(f"\n[Error] {e}\n")