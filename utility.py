import os

import pandas as pd
import pdfplumber
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

import gradio as gr

def extract_data(path):

    flag = 0
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            table = page.extract_tables()
            for pos,r in enumerate(table[0]):
                if flag == 0:
                    p_table = pd.DataFrame(columns=r)
                    flag = 1
                else:
                    if pos==0:
                        continue
                    p_table.loc[len(p_table)] = r

    p_table.set_index('Sr No.', inplace=True)
    return p_table

def embedding(purchase_table, redemption_table):

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    purchase_table['Embedding'] = None
    for index, rows in purchase_table.iterrows():
        purchase_table["Embedding"][index] = embedding_model.encode(rows[1]+' '+rows[2]+' '+rows[3]+' '+rows[4]+' '+rows[6])
    
    redemption_table['Embedding'] = None
    for index, rows in redemption_table.iterrows():
        redemption_table["Embedding"][index] = embedding_model.encode(rows[1]+' '+rows[2]+' '+rows[5])

    purchase_table.to_pickle("Pickle/purchase_table.pkl")
    redemption_table.to_pickle("Pickle/redemption_table.pkl")

def find_similar_doc(prompt, purchase_table, redemption_table):

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embedded_prompt = embedding_model.encode(prompt)

    purchase_similarity_score = []
    redemption_similarity_score = []

    for index, row in purchase_table.iterrows():
        similarity = cosine_similarity(embedded_prompt.reshape(1, -1), 
                                       row['Embedding'].reshape(1, -1))
        
        purchase_similarity_score.append((similarity, index))
    
    purchase_similarity_score = sorted(purchase_similarity_score, key=lambda x: x[0], reverse=True)
    
    for index, row in redemption_table.iterrows():
        similarity = cosine_similarity(embedded_prompt.reshape(1, -1), 
                                       row['Embedding'].reshape(1, -1))
        
        redemption_similarity_score.append((similarity, index))

    redemption_similarity_score = sorted(redemption_similarity_score, key=lambda x: x[0], reverse=True)
    
    return purchase_similarity_score , redemption_similarity_score

def get_doc_for_prompt(prompt, purchase_table, redemption_table):

    purchase_similarity_score, redemption_similarity_score = find_similar_doc(prompt, purchase_table, redemption_table)
    final_prompt = "Use the below data and answer the query. The response should only contain numbers or numbers with denominations.: "
    for i in range(30):
        final_prompt += '\n '+purchase_table.iloc[i][1] +'|'+ purchase_table.iloc[i][2] +'|'+ purchase_table.iloc[i][3] +'|'+ purchase_table.iloc[i][4] +'|'+ purchase_table.iloc[i][7]
        final_prompt += '\n '+purchase_table.iloc[i][1] +'|'+ purchase_table.iloc[i][2] +'|'+ purchase_table.iloc[i][6]

    return final_prompt + "\n Query: " + prompt

def Model(prompt, _):

    purchase_table = pd.read_pickle("Pickle/purchase_table.pkl")
    redemption_table = pd.read_pickle("Pickle/redemption_table.pkl")

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    model = genai.GenerativeModel("gemini-pro")
    prompt = get_doc_for_prompt(prompt, purchase_table, redemption_table)
    prompt = prompt.replace('\n', ' ')
    response = model.generate_content(prompt)

    return response.text

def run_interface(port):

    iface = gr.ChatInterface(fn=Model, title="Get Data")
    iface.launch(share=True, server_port=port)
    


    