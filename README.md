# Content Safety Model

A production-ready **multimodal content safety system** that detects and filters toxic or unsafe content across **text, and images inputs**.  
The system integrates **transformer-based NLP models**, **image captioning**, and layered safety checks for high-accuracy moderation.

---

## 📌 Features

- **Text Safety**: Fine-tuned [DistilBERT](https://arxiv.org/abs/1910.01108) for toxic content classification  
  - Achieved **96% F1-macro** score on the test set.
  - Acts as a **soft classifier** in the safety pipeline.

- **Image Safety**:  
  - Uses **BLIP (Bootstrapping Language-Image Pretraining)** to generate captions.
  - Captions are analyzed for safety using the fine-tuned DistilBERT model.

- **Layered Safety Verification**:  
  - **LLaMA Guard** as a **hard classifier** for strict filtering.  
  - Fine-tuned DistilBERT as a **soft classifier** for nuanced moderation.

- **Deployment**:  
  - End-to-end pipeline deployed with **FastAPI** for real-time inference.

---

## 🛠️ Tech Stack

- **Natural Language Processing**:  
  - [DistilBERT](https://huggingface.co/distilbert-base-uncased) (fine-tuned for toxic content classification)  
  - [LLaMA Guard](https://huggingface.co/meta-llama) for safety classification  

- **Computer Vision**:  
  - [BLIP] for image captioning


- **Deployment**:  
  - [FastAPI] for serving the pipeline  

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/multimodal-content-safety.git
cd multimodal-content-safety
