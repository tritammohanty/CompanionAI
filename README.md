# 🧠 Child-Safe Empathetic AI Companion

A **child-safe, empathetic conversational AI prototype** built on **Mistral-7B (QLoRA)**, enhanced with **emotion awareness, conversational memory, and strict safety filtering**.

This project explores how **parameter-efficient fine-tuning**, **memory augmentation**, and **emotion-conditioned prompting** can improve empathetic responses while maintaining strong safety constraints.

> ⚠️ This is a **research prototype**, not a medical or mental-health professional system.

---

## ✨ Key Features

- ✅ Child-safe response generation  
- 💬 Emotion-aware prompting  
- 🧠 Short-term & long-term conversational memory  
- 🛡 Pre- and post-generation safety filtering  
- ⚡ QLoRA fine-tuning (4-bit, memory efficient)  
- 🧪 Evaluation & ablation studies  

---

## 🏗 Repository Structure
.
├── app/
│ ├── chat_backend.py # Backend pipeline
│ └── companion.py # Chat UI / entrypoint
│
├── src/
│ ├── dataset_utils.py # Helps loading dataset
│ ├── safety_utils.py # Safety filtering logic
│ ├── memory_utils.py # Memory manager
│ ├── emotion_classifier.py # Emotion classification
│ ├── preprocess_and_filter.py # Preproceses and filters dataset
│ ├── model_utils.py # Loads the model
│ └── prompt_utils.py # Prompt construction
│
├── scripts/
│ ├── preprocess.py # Dataset preprocessing
│ ├── train.py # QLoRA training (Accelerate)
│ ├── evaluate.py # Metrics & evaluation
│ └── run_ablation.py # Ablation experiments
│
├── data/
│ ├── raw/ # Original dataset (not tracked)
│ ├── safety/ # Safety folder
│ ├── memory # Stores chat history
│ └── processed/ # Cleaned dataset (not tracked)
│
├── Dockerfile
├── requirements.txt
└── README.md


---

## 🚀 Quick Start
After setting docker run the following commands 
NOTE: ensure to keep port 8000 available for docker
```bash
pip install -r requirements.txt

huggingface-cli login

python scripts/preprocess.py

accelerate config

accelerate launch --config_file configs/default.yaml scripts/train.py

streamlit run app/companion.py --server.port 8000
```

-----
## ⚠️ Disclaimer

This project is for research and educational purposes only.
It is not a replacement for professional mental-health support.

## CONTACT ME
Created by @tritammohanty - feel free to contact me!