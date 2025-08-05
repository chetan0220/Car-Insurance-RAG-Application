import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import os
import requests

def load_paragraphs(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    return paragraphs

def embed_texts(texts, tokenizer, model):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state[:, 0, :].cpu().numpy()
    return embeddings

def call_vextapp_llm(channel_token, api_key, prompt, env="dev"):
    """
    LLM Used: Gemini 2.0 Flash
    This function calls the Vextapp LLM API with the provided channel token and API key
    """
    url = f"https://payload.vextapp.com/hook/Y48P4LP38V/catch/${channel_token}"
    headers = {
        "Content-Type": "application/json",
        "Apikey": f"Api-Key {api_key}"
    }
    data = {
        "payload": prompt,
        "env": env
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        try:
            resp_json = response.json()
            if isinstance(resp_json, dict) and 'text' in resp_json:
                return resp_json['text']
            return resp_json
        except Exception:
            return response.text
    else:
        return f"[Error] Vextapp API returned status code {response.status_code}: {response.text}"

if __name__ == '__main__':
    import torch

    paragraphs = load_paragraphs('car_insurance_faq.txt')

    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Embedding and FAISS index caching
    emb_file = 'paragraph_embeddings.npy'
    faiss_file = 'faiss_index.bin'
    if os.path.exists(emb_file) and os.path.exists(faiss_file):
        paragraph_embeddings = np.load(emb_file)
        index = faiss.read_index(faiss_file)
    else:
        paragraph_embeddings = embed_texts(paragraphs, tokenizer, model)
        np.save(emb_file, paragraph_embeddings)
        dim = paragraph_embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(paragraph_embeddings).astype('float32'))
        faiss.write_index(index, faiss_file)

    VEXTAPP_CHANNEL_TOKEN = os.getenv('VEXTAPP_CHANNEL_TOKEN')
    VEXTAPP_API_KEY = os.getenv('VEXTAPP_API_KEY')


    print('RAG Car Insurance QA System Ready!')
    while True:
        user_q = input('\nAsk a car insurance question (or type "exit"): ')
        if user_q.lower() == 'exit':
            break
        
        user_emb = embed_texts([user_q], tokenizer, model)
        D, I = index.search(np.array(user_emb).astype('float32'), k=3)
        
        retrieved_context = '\n'.join([paragraphs[i] for i in I[0]])
        
        # prompt for LLM
        prompt = f"Context:\n{retrieved_context}\n\nQuestion: {user_q}\nUse the context to answer the question. The first line of the answer should be a direct response to the question, followed by some additional information."
        # prompt = f"WHat is machine learning"
        
        # call Vextapp LLM API
        answer = call_vextapp_llm(VEXTAPP_CHANNEL_TOKEN, VEXTAPP_API_KEY, prompt)
        print(f"\nAnswer: {answer}")
