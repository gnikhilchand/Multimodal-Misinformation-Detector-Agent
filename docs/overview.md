# Multimodal Misinformation Detector

### Purpose
The **Multimodal Misinformation Detector** identifies misinformation in both textual and visual data. The system analyzes user-provided text claims and images to determine their authenticity. For text, it integrates retrieval-based fact-checking and LLM-based reasoning. For images, it leverages a fine-tuned CLIP-based classifier.

### Objective
- Verify factual correctness of user-submitted **text claims** using external evidence and fine-tuned transformers.
- Detect **fake or manipulated images** using a vision-language model.
- Deliver **transparent, explainable outputs** for interpretability.

### Deployed Components

| Component | Description | Platform |
|------------|--------------|-----------|
| **Frontend** | Gradio interface with two tabs (Text Detector, Image Detector) | Hugging Face Spaces |
| **Backend Model Logic** | Python backend (runs inference and retrieval) | Within Space container |
| **Text Model** | `rajyalakshmijampani/fever_finetuned_deberta` | Hugging Face Hub |
| **Image Model** | `rajyalakshmijampani/finetuned_clip` | Hugging Face Hub |
| **LLM Explanation** | `meta-llama/Llama-3.1-8B-Instruct` via `InferenceClient` | Hugging Face Inference API | 
