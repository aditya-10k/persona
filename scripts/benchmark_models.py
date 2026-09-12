"""
Benchmark script comparing embedding models on conversational Hinglish test triplets.
Measures Semantic Discrimination Gap (Delta = Sim(Pos) - Sim(Neg)).
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# Test triplets: (Anchor, Paraphrase/Positive, Unrelated/Negative)
TEST_TRIPLETS = [
    {
        "name": "Meeting / Plans (Hinglish <-> English)",
        "anchor": "bhai kal ka kya scene hai",
        "pos": "bro what is the plan for tomorrow",
        "neg": "machine learning gradient descent algorithm",
    },
    {
        "name": "Slang Agreement (Casual Slang)",
        "anchor": "sahi hai yaar chal done",
        "pos": "sounds good bro let's do it",
        "neg": "federal reserve interest rate hike",
    },
    {
        "name": "Casual Venting (Emotion / Low Mood)",
        "anchor": "bhai bohot thak gaya hu aaj dimag kharab hai",
        "pos": "bro completely exhausted today feeling super low",
        "neg": "photosynthesis in green plants",
    },
    {
        "name": "Code / Tech Question",
        "anchor": "python me ye error kyu aa raha hai",
        "pos": "why is this bug showing up in python",
        "neg": "biryani recipe step by step",
    },
]

MODELS = [
    ("paraphrase-multilingual-MiniLM-L12-v2", "paraphrase-multilingual-MiniLM-L12-v2"),
    ("all-MiniLM-L6-v2", "all-MiniLM-L6-v2"),
]

def evaluate_model(name, model_id):
    print(f"\n{'='*55}\nEvaluating: {name}\n{'='*55}")
    model = SentenceTransformer(model_id)
    gaps = []
    
    for item in TEST_TRIPLETS:
        vecs = model.encode([item["anchor"], item["pos"], item["neg"]], normalize_embeddings=True)
        sim_pos = float(vecs[0] @ vecs[1])
        sim_neg = float(vecs[0] @ vecs[2])
        delta = sim_pos - sim_neg
        gaps.append(delta)
        print(f"[{item['name']}]")
        print(f"  Anchor:   '{item['anchor']}'")
        print(f"  Positive: '{item['pos']}' -> Sim: {sim_pos:.4f}")
        print(f"  Negative: '{item['neg']}' -> Sim: {sim_neg:.4f}")
        print(f"  --> Contrastive Gap (Delta): {delta:+.4f}\n")
        
    avg_gap = np.mean(gaps)
    print(f"AVERAGE CONTRASTIVE GAP for {name}: {avg_gap:.4f}")
    return avg_gap

if __name__ == "__main__":
    for name, mid in MODELS:
        evaluate_model(name, mid)
