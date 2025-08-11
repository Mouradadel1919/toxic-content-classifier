import uvicorn
from fastapi import FastAPI
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from fastapi.responses import JSONResponse
import io

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import transformers
import tensorflow as tf
from nltk.corpus import stopwords
from transformers import DistilBertTokenizer, DistilBertModel, AutoTokenizer, AutoModel, BlipProcessor, BlipForConditionalGeneration, AutoModelForCausalLM, pipeline
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models, regularizers
from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
import re
import nltk
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

dataset = pd.read_csv("/content/cellula toxic data.csv")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# 1. Cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove non-letters
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# 2. Lemmatization function
def lemmatize_text(text):
    tokens = word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(token) for token in tokens])

# 3. Full pipeline
def text_pipeline(text):
    cleaned = clean_text(text)
    lemmatized = lemmatize_text(cleaned)
    return lemmatized


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
bert_model = DistilBertModel.from_pretrained(model_ckpt, num_labels=9).to(device)

ann_model = load_model("/content/model_ann.keras")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/LlamaGuard-7b", use_auth_token=token)
llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/LlamaGuard-7b", use_auth_token=token, device_map="auto", torch_dtype=torch.float16)


def SoftClassifier(ann_model, bert_model, tokenizer, text):
    text = text_pipeline(text)

    tokenized_sentence = tokenizer(text, return_special_tokens_mask= True, return_tensors='pt', padding=True, truncation=True).to(device)
    input_ids = tokenized_sentence["input_ids"].to(device)
    attention_mask = tokenized_sentence["attention_mask"].to(device)
    special_tokens_mask = tokenized_sentence["special_tokens_mask"].to(device)

    with torch.no_grad():
        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state


    pooled_embeddings = []
    for i in range(hidden_states.size(0)):
        non_special = ~special_tokens_mask[i].bool()
        clean_embed = hidden_states[i][non_special]  # [num_tokens, 768]
        mean_pooled = clean_embed.mean(dim=0).cpu().numpy()  # [768]
        pooled_embeddings.append(mean_pooled)


    lab2str = {
    0: 'Safe',
    1: 'unsafe',
    2: 'Suicide & Self-Harm',
    3: 'Unknown S-Type',
    4: 'Elections',
    5: 'Violent Crimes',
    6: 'Sex-Related Crimes',
    7: 'Non-Violent Crimes',
    8: 'Child Sexual Exploitation'
     }

    test = pooled_embeddings[0].reshape(1, 768)
    y_pred= ann_model.predict(test)
    y_pred = np.argmax(y_pred, axis=1)

    return lab2str[y_pred[0]]

def image_captioning(blip_model, processor, image):

    # Process the image
    inputs = processor(images=image, return_tensors="pt")
    outputs = blip_model.generate(**inputs)

    caption = processor.decode(outputs[0], skip_special_tokens=True)

    return caption


def classify_text(text):
    pipe = pipeline("text-generation", model=llama_model, tokenizer=llama_tokenizer)

    def get_prompt(input_text):
        return f"""<s>[INST] <<SYS>>
                    You are a text safety classifier. You will classify the following user input as either SAFE or UNSAFE.
                    <</SYS>>

                    Text: {input_text}
                    Classification: [/INST]"""


    response = pipe(get_prompt(text), max_new_tokens=10, return_full_text=False)[0]['generated_text']
    hard = "None"
    soft = "None"
    if 'unsafe' in response:
        hard = f'Hard Classifier : {"unsafe"}'
        soft = f'Soft Classifier :  {SoftClassifier(ann_model, bert_model, tokenizer, text)}'

    elif 'safe' in response:
        hard = f'Hard Classifier : {"safe"}'
        soft = f'Soft Classifier :  {SoftClassifier(ann_model, bert_model, tokenizer, text)}'

    else:
        hard = 'unknown'
        soft = 'unknown'
    print (hard, soft)
    return hard, soft


from pyngrok import ngrok
import nest_asyncio

app = FastAPI()
templates = Jinja2Templates(directory="/content/drive/MyDrive/app/templates")
app.mount("/static", StaticFiles(directory="/content/drive/MyDrive/app/static"), name="static")

@app.get('/', response_class=HTMLResponse)
@app.get('/home', response_class=HTMLResponse)

async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



@app.get("/predict", response_class=HTMLResponse)
async def get_predict(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request})


@app.post("/predict")
async def post_predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    caption = image_captioning(blip_model, processor, image)
    hard, soft = classify_text(caption)
    return {"caption": caption, "hard": hard, "soft":soft}





@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

# Apply asyncio patch
nest_asyncio.apply()

# Open a tunnel on port 8000
public_url = ngrok.connect(8000)
print("Public URL:", public_url)

# Run the app
uvicorn.run(app, port=8000)