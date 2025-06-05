import streamlit as st
st.set_page_config(page_title="Semantic Product Search", layout="centered")  
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

PRODUCT_CSV_PATH = "D:/PROjects/nlp project/cleaned_merged_products.csv"
EMBEDDING_PATH = "D:/Opera_downloads/product_embeddings.npy"
MODEL_PATH = "D:/PROjects/nlp project/saved_query_model"  


@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_PATH)

@st.cache_data
def load_data():
    df = pd.read_csv(PRODUCT_CSV_PATH)
    embeddings = np.load(EMBEDDING_PATH)
    embeddings_tensor = torch.tensor(embeddings)
    return df, embeddings_tensor

model = load_model()
df, product_embeddings = load_data()

device = torch.device('cpu')
product_embeddings = product_embeddings.to(device)
model = model.to(device)


def semantic_search(query, top_k=5):
    query_embedding = model.encode(query, convert_to_tensor=True).to(device)

    cosine_scores = util.cos_sim(query_embedding, product_embeddings)[0]

    top_results = torch.topk(cosine_scores, k=top_k * 5)  # Fetch more to allow for duplicates filtering

    results = []
    seen_ids = set()

    for score, idx in zip(top_results.values, top_results.indices):
        product_info = df.iloc[idx.item()]
        pid = product_info['product_id']

        if pid not in seen_ids:
            results.append((score.item(), product_info))
            seen_ids.add(pid)

        if len(results) >= top_k:
            break

    return results




st.title("🛍️ Semantic Product Search")

user_query = st.text_input("Enter a product query:", placeholder="e.g. brown chair")

top_k = st.slider("Number of results to show", min_value=1, max_value=20, value=5)

if st.button("Search") and user_query:
    with st.spinner("Searching..."):
        results = semantic_search(user_query, top_k=top_k)

    st.subheader("🔎 Results")
    for score, product in results:
        st.markdown(f"""
        **Product ID**: {product.get('product_id', 'N/A')}  
        **Name**: {product.get('product_name', 'N/A')}  
        **Description**: {product.get('product_description', 'N/A')}  
        ---""")
