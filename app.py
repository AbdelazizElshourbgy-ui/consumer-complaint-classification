"""
Consumer Complaint Classifier - Gradio app

Loads the fine-tuned DistilBERT model saved by the notebook (best_model/ folder)
and returns the predicted complaint category with confidence scores.

Run:
    pip install gradio transformers torch
    python app.py
"""
import os

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------- Config ----------------
# Folder saved by the notebook (trainer.save_model + tokenizer.save_pretrained).
# Override with: MODEL_DIR=/path/to/best_model python app.py
CANDIDATE_DIRS = [
    os.environ.get("MODEL_DIR", ""),
    "best_model",
    "/kaggle/working/best_model",
]
MODEL_DIR = next((d for d in CANDIDATE_DIRS if d and os.path.isdir(d)), None)
if MODEL_DIR is None:
    raise FileNotFoundError(
        "Model folder not found. Put the 'best_model' folder next to app.py "
        "or set the MODEL_DIR environment variable."
    )

MAX_LEN = 256
TOP_K = 5

# ---------------- Load model ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()
id2label = model.config.id2label  # saved inside config.json


# ---------------- Inference ----------------
def predict(text: str):
    if not text or not text.strip():
        return {}
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=MAX_LEN
    ).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0].cpu().numpy()
    top = probs.argsort()[::-1][:TOP_K]
    return {id2label[int(i)]: float(probs[i]) for i in top}


# ---------------- UI ----------------
demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        lines=6,
        label="Complaint narrative",
        placeholder="Type or paste a customer complaint...",
    ),
    outputs=gr.Label(num_top_classes=TOP_K, label="Predicted category"),
    title="Consumer Complaint Classifier",
    description="Enter a complaint and get the predicted category with confidence scores.",
    examples=[
        ["I was charged twice on my credit card and the bank refuses to refund me."],
        ["A debt collector keeps calling me about a debt that is not mine."],
        ["My mortgage servicer lost my payment and charged me late fees."],
    ],
)

if __name__ == "__main__":
    demo.launch()
