# Multimodal Content Safety Pipeline

A production-ready **multimodal content safety system** that detects and filters toxic or unsafe content across **text, images, and voice inputs**.  
The system integrates **transformer-based NLP models**, **image captioning**, and layered safety checks for high-accuracy moderation.

---

## 📌 Features

- **Text Safety**: Fine-tuned [DistilBERT](https://arxiv.org/abs/1910.01108) for toxic content classification  
  - Achieved **96% F1-macro** score on the test set.
  - Acts as a **soft classifier** in the safety pipeline.

- **Image Safety**:  
  - Uses **BLIP (Bootstrapping Language-Image Pretraining)** to generate captions.
  - Captions are analyzed for safety using the fine-tuned DistilBERT model.

- **Voice Safety**:  
  - Transcribes audio to text via ASR (Automatic Speech Recognition).
  - Processes the transcript with the text safety models.

- **Layered Safety Verification**:  
  - **LLaMA Guard** as a **hard classifier** for strict filtering.  
  - Fine-tuned DistilBERT as a **soft classifier** for nuanced moderation.

- **Deployment**:  
  - End-to-end pipeline deployed with **FastAPI** for real-time inference.
  - Fully containerized and ready for production environments.

---

## 🛠️ Tech Stack

- **Natural Language Processing**:  
  - [DistilBERT](https://huggingface.co/distilbert-base-uncased) (fine-tuned for toxic content classification)  
  - [LLaMA Guard](https://huggingface.co/meta-llama) for safety classification  

- **Computer Vision**:  
  - [BLIP](https://github.com/salesforce/BLIP) for image captioning

- **Speech Processing**:  
  - Automatic Speech Recognition (ASR) for voice input transcription

- **Deployment**:  
  - [FastAPI](https://fastapi.tiangolo.com/) for serving the pipeline  
  - [Docker](https://www.docker.com/) for containerization  

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/multimodal-content-safety.git
cd multimodal-content-safety
