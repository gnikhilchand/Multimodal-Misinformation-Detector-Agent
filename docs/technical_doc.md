# Technical Documentation

## 1. Environment Setup

**Python Version:** 3.10+  
**Hardware Requirements:**
- Minimum: 4GB RAM (for image inference only)
- Recommended: GPU with CUDA for faster processing

**Dependencies:**
```bash
torch
torchvision
torchaudio
transformers
sentence-transformers
gradio
huggingface-hub
tavily
wikipedia
wikipedia-api
scikit-learn
requests
```
**Installation Steps**
```bash
git clone https://huggingface.co/spaces/group9-dsailab/multimodal_misinfo_detector.git
cd multimodal_misinfo_detector
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```


## 2. Data Pipeline

**Sources**
- **Text Claims**: Derived from the FEVER dataset (fact verification dataset).
- **Evidence Sources**:  
    - Google Fact Check API – verified claims from fact-checking organizations.
    - Tavily API – retrieves relevant web evidence.
    - Wikipedia – general background information.
- **Preprocessing**:
    - Tokenization using AutoTokenizer (DeBERTa).
    - Sentence segmentation and filtering (regex-based).
    - Embedding and ranking using SentenceTransformer(all-MiniLM-L6-v2).
- **Licensing**:
    - FEVER dataset: CC BY-SA 4.0.
    - Wikipedia content: CC BY-SA 3.0.
    - External API usage per provider’s terms.

## 3. Model Architecture

**Text Classifier**
- **Base Model**: DeBERTa-v3-base
- **Fine-tuned Model**: rajyalakshmijampani/fever_finetuned_deberta
- **Task**: Binary classification (REAL / FAKE)
- **Max Length**: 512 tokens

**Image Classifier**

- Custom `CLIPImageClassifier` built on `openai/clip-vit-base-patch32`
- Vision Encoder (CLIP) → Linear(1024 → 256) → ReLU → Dropout(0.5) → Linear(256 → 1) + Sigmoid  
- **Output**: Probability (0–1)
- **Threshold**: >0.5 = Fake; otherwise Real

**Explanation Model**
- **LLM**: `meta-llama/Llama-3.1-8B-Instruct`
- **API**: Hugging Face InferenceClient
- **Output**: Structured JSON with verdict, explanation, confidence

## 4. Training Summary

**Text Model**

- **Base**: DeBERTa-v3-base
- **Dataset**: FEVER (Claim-Evidence pairs)
- **Optimizer**: Adam (lr=2e-5)
- **Epochs**: 3
- **Loss**: CrossEntropyLoss
- **Accuracy**: 95%
- **F1 Score**: ~0.95

**Image Model**

- **Base**: CLIP ViT-B/32
- **Optimizer**: Adam (lr=1e-4)
- **Epochs**: 3
- **Loss**: Binary CrossEntropy
- **Accuracy**: ~96%

## 5. Inference Pipeline

**Text Inference Flow**

```
def classify_text(claim, hf_token, tavily_key, google_key):
    classifier = get_text_classifier()
    evidences = get_evidence_sentences(claim)
    evidence_text = " ".join(evidences)
    result = classifier(f"claim: {claim} evidence: {evidence_text}")
    label = "REAL" if result[0]['label'] == "LABEL_0" else "FAKE"
    return label
```

**Image Inference Flow**

```
def classify_image(image):
    inputs = image_processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"]
    output = classifier(inputs)
    label = "Fake" if output.item() > 0.5 else "Real"
    return label
```
## 6. Deployment Details

| Parameter           | Description                                       |
| ------------------- | ------------------------------------------------- |
| **Platform**        | Hugging Face Spaces                               |
| **Frontend**        | Gradio                                            |
| **Backend**         | Python (single app.py)                            |
| **Models**          | Loaded via Hugging Face Hub                       |
| **Inference**       | Local execution or via Hugging Face Inference API |
| **Endpoint Access** | UI at space URL (no REST API exposed)             |

## 7. System Design Considerations

- **Scalability**: Modular design allows future replacement of evidence retrieval or LLM components.
- **Reusability**: Functions like _from_tavily, _from_google, _rank_sentences are independent utilities.
- **Extensibility**: The same pipeline can support new evidence APIs or visual models.
- **Data Flow**: Two independent inference routes (text, image) unified under one Gradio interface.
- **Security**: Sensitive API keys handled as masked UI inputs; no server storage.

## 8. Error Handling & Monitoring

- **Graceful Failures**: _safe_call() wrapper ensures that API failures (Google, Tavily, Wikipedia) don’t crash inference.
- **JSON Validation**: LLM output parsing uses json.loads() with fallback to raw text.
- **Logging**: Console warnings for retrieval errors or JSON parsing issues.
- **Latency Consideration**: HF spaces default cached models reduce repeated download time.

## 9. Reproducibility Checklist

| Item           | Description                                                                         |
| -------------- | ----------------------------------------------------------------------------------- |
| Python Version | 3.10+                                                                               |
| Random Seed    | 42                                                                                  |
| Checkpoints    | `rajyalakshmijampani/fever_finetuned_deberta`, `rajyalakshmijampani/finetuned_clip` |
| Config         | All hyperparameters stored in training notebooks (Milestone 4)                      |
| Data           | FEVER, image dataset (custom social dataset, CC licensed)                           |
| Scripts        | `app.py`, training notebooks (available in repo)                                    |

## 10. Maintenance notes

**Future Enhancements**

- **Model Improvements**: Experiment with larger LLMs or multimodal transformers.
- **Additional Modalities**: Extend to video or audio misinformation detection.
- **API Exposure**: Develop REST API endpoints for programmatic access.
- **User Feedback Loop**: Integrate user feedback to improve model accuracy over time.
- **Monitoring Dashboard**: Implement real-time monitoring of usage and performance metrics.
- **Automated Testing**: Set up unit and integration tests for core functions and models.

**Known Limitations**

- **LLM Dependence**: Explanation quality relies on LLM capabilities and may vary.
- **Evidence Coverage**: Retrieval APIs may not always find relevant evidence, impacting text classification.
- **Language Dependence**: Limited to English claims and Wikipedia evidence.
- **Image Variability**: Performance may degrade on out-of-distribution or highly manipulated images.

## 11. Contact Information

Group 9 - DSAI Lab
- Rajya Lakshmi Jampani (22f3002986@ds.study.iitm.ac.in)
- Faiz Malik (23f2004839@ds.study.iitm.ac.in) 
- Sachin Singh (21f1003251@ds.study.iitm.ac.in)
- G. Nikhil Chand (21f1003825@ds.study.iitm.ac.in) 
- C. Glenn Varshan (22f2000961@ds.study.iitm.ac.in) 
 
 
 
 
 
 


