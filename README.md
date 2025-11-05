# Multimodal Misinformation Detector Agent  

This repository contains all materials developed for the **Multimodal Misinformation Detector Agent** project, which aims to assess the credibility of textual and visual information using deep learning and natural language understanding techniques. The system integrates evidence retrieval, transformer-based textual inference, and image authenticity detection to generate an interpretable credibility assessment.  

## Repository Organization  
```
├── code/  
|   ├──EDA_Preprocessing
│   |   ├── fever_train_ready.jsonl
│   │   ├── image_eda.ipynb              
│   │   ├── text_eda_liar_dataset.ipynb         
│   |   ├── text_preprocessing.ipynb     
│   |  
│   ├──Finetuning
│   │   ├── clip_finetuning.ipynb
|   │   ├── deberta_finetuning.ipynb
|   |   ├── Gemma3-270m_finetuning.ipynb
|   │        
│   ├── text_benchmarking.ipynb      
│                         
├── Milestone1_Report.pdf            
├── Milestone2_Report.pdf            
├── Milestone3_Report.pdf            
├── Milestone4_Report.pdf  
├── artifacts.txt            
└── README.md
```
  
## Guidance for Reviewers  

1. **Sequential Reading of Reports**  
   Review the milestone reports in order. Each document is self-contained and collectively describes the complete research process — from conceptualization and dataset preparation to model development and training analysis.

2. **Code Exploration**  
   The `/code/` directory contains Jupyter notebooks that correspond to the experimental phases EDA + Preprocessing, Fine-tuning, and Benchmarking.
    
   Each notebook includes structured markdown cells detailing objectives, methodology, and observations.

3. **Dataset and Artifacts**  
   - The file `fever_train_ready.jsonl` contains the finalized dataset used for DeBERTa fine-tuning.  
   - `artifacts.txt` documents saved checkpoints, evaluation logs, and configurations for reproducibility.
  
## Key Components  

- **Text Credibility Assessment:** Fine-tuned *microsoft/deberta-v3-base* on the FEVER dataset for claim-evidence verification.  
- **Image Authenticity Detection:** Fine-tuned *openai/clip-vit-base-patch32* to classify real versus manipulated images.  
- **Natural-Language Explanation:** Generated through *mistralai/Mistral-7B-Instruct-v0.2* to provide human-readable reasoning.  
- **Evidence Retrieval:** Integrated sources include Google Fact Check, Tavily API, and Wikimedia.


## Summary  

The repository is designed for transparent evaluation of multimodal misinformation detection workflows. Each milestone builds incrementally towards the integrated system, demonstrating both the theoretical grounding and the empirical results of model development and optimization.  

We welcome feedback and suggestions to further enhance the system's capabilities and robustness.




