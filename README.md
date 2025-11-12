# Multimodal Misinformation Detector Agent  

This repository contains all materials developed for the **Multimodal Misinformation Detector Agent** project, which aims to assess the credibility of textual and visual information using deep learning and natural language understanding techniques. The system integrates evidence retrieval, transformer-based textual inference, and image authenticity detection to generate an interpretable credibility assessment.  

## Repository Organization  
```
MULTIMODAL-MISINFORMATION-DETECTOR-AGENT/  
│  
├── app/                             # Application code  
│   └── app.py  
│  
├── data/                            # Data files (links, processed)  
│   ├── dataset_links.txt  
│   └── fever_train_ready.json  
│  
├── docs/                            # Documentation for users/developers  
│   ├── overview.md  
│   ├── technical_doc.md  
│   ├── user_guide.md  
|  
├── models/                          # Model weights (links)  
│   └── model_links.txt  
│  
├── notebooks/                       # Jupyter notebooks for experiments  
│   ├── EDA_Preprocessing/  
│   │   ├── image_eda.ipynb  
│   │   ├── text_eda.ipynb  
│   │   └── text_preprocessing.ipynb  
│   │  
│   ├── Finetuning/  
│   │   ├── clip_convnextv2_maxvit_finetuning.ipynb   
│   │   ├── clip_finetuning.ipynb  
|   |   ├── clip_image_prediction.py  
|   |   ├── clip_video_prediction.py  
|   |   ├── clip_zero_shot_image_prediction.py  
│   │   ├── deberta_finetuning.ipynb  
│   │   └── gemma3_270m_finetuning.ipynb  
│   │  
│   └── text_benchmarking.ipynb  
│  
├── reports/                         # Milestone & evaluation reports  
│   ├── Milestone1_Report.pdf  
│   ├── Milestone2_Report.pdf  
│   ├── Milestone3_Report.pdf  
│   ├── Milestone4_Report.pdf  
│   ├── Milestone5_Report.pdf  
│   └── Milestone6_Report.pdf  
│  
├── requirements.txt                 # Python dependencies  
├── README.md                        # High-level project overview
```
  
## Guidance for Reviewers  

1. **Sequential Reading of Reports**  
   The `reports/` directory contains milestone reports (1 through 6) that document the project's progression. It is recommended to read these reports in order to understand the development stages, challenges faced, and solutions implemented.

2. **Code Exploration**  
   The `notebooks/` directory houses Jupyter notebooks that detail exploratory data analysis (EDA), preprocessing steps, and model fine-tuning procedures. Each notebook includes structured markdown cells detailing objectives, methodology, and observations.

3. **Dataset and Artifacts**  
   The `data/` and `models/` directories contain links to datasets and pre-trained model weights used throughout the project. Reviewers can follow these links to access the necessary resources for replication.

4. **Documentation**  
   The `docs/` directory provides comprehensive documentation, including an overview of the system, a user guide for non-technical users, and this technical documentation for developers.

5. **Application Code**  
   The `app/` directory contains the main application code (`app.py`) that integrates all components into a functional system. Reviewers can examine this code to understand how different modules interact.

6. **Project Setup**  
   The `requirements.txt` file lists all Python dependencies required to run the application. Reviewers can use this file to set up their environment for testing and evaluation.
  
## Key Components  

- **Text Credibility Assessment:** Fine-tuned *microsoft/deberta-v3-base* on the FEVER dataset for claim-evidence verification.  
- **Image Authenticity Detection:** Fine-tuned *openai/clip-vit-base-patch32* to classify real versus manipulated images.  
- **Natural-Language Explanation:** Generated through *meta-llama/Llama-3.1-8B-Instruct* to provide human-readable reasoning.  
- **Evidence Retrieval:** Integrated sources include Google Fact Check, Tavily API, and Wikimedia.
- **User Interface:** Built with Gradio and deployed on Hugging Face Spaces for accessibility.


## Summary  

The repository is designed for transparent evaluation of multimodal misinformation detection workflows. Each milestone builds incrementally towards the integrated system, demonstrating both the theoretical grounding and the empirical results of model development and optimization.  

We welcome feedback and suggestions to further enhance the system's capabilities and robustness.

- Rajya Lakshmi Jampani (22f3002986@ds.study.iitm.ac.in)
- Faiz Malik (23f2004839@ds.study.iitm.ac.in) 
- Sachin Singh (21f1003251@ds.study.iitm.ac.in)
- G. Nikhil Chand (21f1003825@ds.study.iitm.ac.in) 
- C. Glenn Varshan (22f2000961@ds.study.iitm.ac.in) 




