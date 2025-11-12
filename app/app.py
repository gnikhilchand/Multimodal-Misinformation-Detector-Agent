import os
import torch
from torch import nn
import json
import requests
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, CLIPProcessor, CLIPModel
from collections import OrderedDict
import wikipedia
import wikipediaapi
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tavily import TavilyClient
from huggingface_hub import InferenceClient, hf_hub_download

class CLIPImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.classifier = nn.Sequential(
            nn.Linear(self.clip.config.vision_config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, pixel_values):
        feats = self.clip.vision_model(pixel_values=pixel_values).pooler_output
        return self.classifier(feats)

text_classifier = None
image_classifier = None
TAVILY_KEY = None
GOOGLE_KEY = None
HF_TOKEN = None

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
explain_model = "meta-llama/Llama-3.1-8B-Instruct"
text_model = "rajyalakshmijampani/fever_finetuned_deberta"
image_model = "rajyalakshmijampani/finetuned_clip"
wiki = wikipediaapi.Wikipedia(language='en', user_agent='fact-checker/1.0')
image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_text_classifier():
    global text_classifier
    if text_classifier is None:
        tokenizer = AutoTokenizer.from_pretrained(text_model)
        seq_clf = AutoModelForSequenceClassification.from_pretrained(text_model)
        text_classifier = pipeline("text-classification", model=seq_clf, tokenizer=tokenizer)
    return text_classifier

def get_image_classifier():
    global image_classifier, image_model
    filename = "finetuned_clip.pth"
    if image_classifier is None:
        model_path = hf_hub_download(repo_id=image_model, filename=filename)
        image_classifier = CLIPImageClassifier()
        state = torch.load(model_path, map_location="cpu",weights_only=False)
        clean_state = OrderedDict(
            (k[7:], v) if k.startswith("module.") else (k, v)
            for k, v in state.items()
        )
        image_classifier.load_state_dict(clean_state, strict=False)
        image_classifier.eval()
        return image_classifier
    
    return image_classifier

def _rank_sentences(claim, sentences, top_k=4):
    if not sentences: return []
    emb_c = embed_model.encode([claim])
    emb_s = embed_model.encode(sentences)
    sims = cosine_similarity(emb_c, emb_s)[0]

    claim_tokens = set(re.findall(r'\w+', claim.lower()))
    scored = []
    for s, sim in zip(sentences, sims):
        overlap = len(claim_tokens.intersection(set(re.findall(r'\w+', s.lower()))))
        scored.append((s, sim + 0.01 * overlap))  # small lexical boost
    ranked = [s for s, _ in sorted(scored, key=lambda x: x[1], reverse=True)]
    return ranked[:top_k]

def _split_sentences(text):
    sents = re.split(r'(?<=[.!?])\s+', text)
    clean = []
    for s in sents:
        s = s.strip()
        if 15 < len(s) < 350 and not s.lower().startswith(("see also", "references", "external links")):
            clean.append(s)
    return clean

def _safe_call(func, claim):
    try:
        return func(claim)
    except Exception as e:
        print(f"[WARN] {func.__name__} failed: {e}")
        return []

def _from_google(claim):
    global GOOGLE_KEY
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    r = requests.get(url, params={"query": claim, "key": GOOGLE_KEY, "pageSize": 2}).json()
    claims = r.get("claims", [])
    evid = []
    for c in claims:
        rev = c.get("claimReview", [])
        if rev:
            rating = rev[0].get("textualRating", "")
            site = rev[0].get("publisher", {}).get("name", "")
            evid.append(f"{site} rated this claim as {rating}.")
    return evid[:3]

def _from_tavily(claim):
    global TAVILY_KEY
    tavily = TavilyClient(api_key=TAVILY_KEY)
    try:
        results = tavily.search(claim).get("results", [])
        sents = []
        for r in results:
            for s in _split_sentences(r.get("content", "")):
                if not any(x in s.lower() for x in ["video game", "film", "fiction"]):
                    sents.append(s)
        return _rank_sentences(claim, sents, 4)
    except Exception:
        return []
    
def _from_wiki(claim):
    try:
        titles = wikipedia.search(claim, results=3)
        sents = []
        for t in titles:
            page = wiki.page(t)
            if not page.exists(): continue
            text = page.text[:5000]  # extend a bit
            for s in _split_sentences(text):
                if not any(x in s.lower() for x in ["video game", "fiction", "film"]):
                    sents.append(s)
        return _rank_sentences(claim, sents, 4)
    except Exception as e:
        print(f"[WARN] _from_wiki failed: {e}")
        return []

def get_evidence_sentences(claim, k=3):
    evid = _safe_call(_from_google, claim)
    if len(evid) >= k: return evid[:k]
    evid += _safe_call(_from_tavily, claim)
    if len(evid) >= k: return evid[:k]
    evid += _safe_call(_from_wiki, claim)
    evid = [e for e in evid if len(e.strip()) > 10]
    return (evid or ["Error: No relevant evidence found."])[:k]

# ---Text Classification Function ---
def classify_text(claim, hf_token, tavily_key, google_key):

    global HF_TOKEN, TAVILY_KEY, GOOGLE_KEY
    HF_TOKEN = hf_token.strip()
    TAVILY_KEY = tavily_key.strip()
    GOOGLE_KEY = google_key.strip()

    claim=claim.lower().strip()
    classifier = get_text_classifier()
    evidences = get_evidence_sentences(claim)    
    evidence_text = " ".join(evidences).lower().strip()

    # Step 1: FEVER classification
    text = f"claim: {claim} evidence: {evidence_text}"
    result = classifier(text, truncation=True, max_length=512, return_all_scores=True)[0]
    top_label = sorted(result, key=lambda x: x["score"], reverse=True)[0]["label"]
    label_str = "REAL" if top_label == "LABEL_0" else "FAKE"
    print(f"[INFO] Model Classified {claim} as {label_str}")

    # Step 2: Mistral explanation generation
    prompt = f"""
            You are a reliable fact-checking assistant.

            User's statement: "{claim}"

            Information you have received (use this for reasoning, but do not mention or list it directly):
            {chr(10).join(f"- {e}" for e in evidences)}

            The system’s current assessment is that the claim is: "{label_str}".

            Now, carefully evaluate the statement and the assessment. You may disagree with the system if the evidences clearly contradict the claim.
            Write your reasoning and return it STRICTLY as a JSON object with the following fields:
            {{
                "verdict": "Real / Fake / Uncertain",
                "explanation": "3–5 natural sentences explaining what makes the claim true or fake or uncertain. 
                                Do NOT mention words like 'evidence', 'sources', or 'provided information'. 
                                Instead, explain the reasoning naturally as if you are telling it from general knowledge.",",
                "confidence": "Low / Medium / High 
                               Decide this depending on how strong the evidences are, how clear the reasoning is,
                               and how certain you are about your verdict."
            }}
            Do NOT include anything outside the JSON. Use plain text, no Markdown. Be concise and to the point.
"""
    messages = [
        {"role": "system", "content": "You are a reliable fact-checking assistant."},
        {"role": "user", "content": prompt},
    ]

    inf_client = InferenceClient(token=HF_TOKEN)
    completion = inf_client.chat_completion( model=explain_model, messages=messages, max_tokens=256, temperature=0.3)
    raw_response = completion.choices[0].message.content.strip()

    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError:
        print("[WARN] Could not parse JSON, returning raw text")
        return raw_response
    
    formatted_output = f"""**Prediction:** The claim is {data.get('verdict', 'N/A')}.

**Explanation:**
{data.get('explanation', 'No explanation available.')}

**Confidence:** {data.get('confidence', 'N/A')}."""  
    
    return formatted_output.strip()


# ---- Image classification Function ----
def classify_image(image):
    global image_processor
    classifier = get_image_classifier()
    try:
        inputs = image_processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"]
        with torch.no_grad():
            output = classifier(inputs)
        p = output.item()
        label = "Fake" if p > 0.5 else "Real"
        return f"**Prediction:** {label}\n**Confidence score:** {p:.2f}"
    except Exception as e:
        return f"Error: {e}"

# -------------------
# UI Layout (Gradio)
# -------------------
with gr.Blocks() as demo:
    gr.Markdown("# Multimodal Misinformation Detector")

    with gr.Tab("Text Detector"):
        with gr.Row(): 
            with gr.Column(scale=3): #  Left half — main inputs
                claim = gr.Textbox(label="Enter Claim")
                text_button = gr.Button("Classify Claim", interactive=False) # Disable until tokens provided
                text_output = gr.Markdown( label="Model Output", value="Results will appear here...")

            
            with gr.Column(scale=1):  # Right half — user token inputs
                gr.Markdown("## Enter your API keys")
                hf_token = gr.Textbox(label="Hugging Face Token 🔴", type="password")
                tavily_key = gr.Textbox(label="Tavily API Key 🔴", type="password")
                google_key = gr.Textbox(label="Google Fact Check API Key 🔴", type="password")
            
        # Enable button when all fields filled
        def enable_button(hf, tavily, google):
            ready = bool(hf and tavily and google)
            return gr.update(interactive=ready)
        
        hf_token.change(enable_button, inputs=[hf_token, tavily_key, google_key], outputs=text_button)
        tavily_key.change(enable_button, inputs=[hf_token, tavily_key, google_key], outputs=text_button)
        google_key.change(enable_button, inputs=[hf_token, tavily_key, google_key], outputs=text_button)

        # Click handler (include all token inputs)
        text_button.click(classify_text,
                          inputs=[claim, hf_token, tavily_key, google_key],
                          outputs=text_output)
        

    with gr.Tab("Image Detector"):
        img_input = gr.Image(type="pil", label="Upload Image")
        img_button = gr.Button("Classify Image")
        img_output = gr.Markdown(label="Model Output", value="Results will appear here...")
        
        img_button.click(classify_image, inputs=img_input, outputs=img_output)

demo.launch()
