import os
import pandas as pd
import numpy as np
from openai import AzureOpenAI
import dotenv
import chromadb
from chromadb.config import Settings

dotenv.load_dotenv()

df = pd.DataFrame(columns=['path', 'text'])


# splitting our data into chunks
data_paths= ["data/dayron.md","data/dayron_cv.md"]

for path in data_paths:
    with open(path, 'r', encoding='utf-8') as file:
        file_content = file.read()

    # Append the file path and text to the DataFrame
    df = pd.concat([df, pd.DataFrame([{'path': path, 'text': file_content}])], ignore_index=True)
    #df = df.append({'path': path, 'text': file_content}, ignore_index=True)



def split_text(text, max_length, min_length):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(' '.join(current_chunk)) < max_length and len(' '.join(current_chunk)) > min_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = []

    # If the last chunk didn't reach the minimum length, add it anyway
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

# Assuming analyzed_df is a pandas DataFrame and 'output_content' is a column in that DataFrame
splitted_df = df.copy()
splitted_df['chunks'] = splitted_df['text'].apply(lambda x: split_text(x, 400, 100))

# Assuming 'chunks' is a column of lists in the DataFrame splitted_df, we will split the chunks into different rows
flattened_df = splitted_df.explode('chunks')


client = AzureOpenAI(
    azure_endpoint="https://hatchworksai.openai.azure.com/",
    api_key=os.environ['AZURE_OPENAI_KEY'],
    api_version="2023-10-01-preview", 
)

def create_embeddings(text, model="ada-02"):
    # Create embeddings for each document chunk
    embeddings = client.embeddings.create(input = text, model=model).data[0].embedding
    return embeddings

chroma_client = chromadb.PersistentClient(path="./DB/")

# Nombre de la colección
collection_name = "embeddings_collection"

# Verificar si la colección existe
collections = chroma_client.list_collections()
collection_names = [col.name for col in collections]
if collection_name not in collection_names:
    # Crear la colección si no existe
    chroma_client.create_collection(name=collection_name)

# Obtener la colección (ya sea recién creada o existente)
collection = chroma_client.get_collection(name=collection_name)

embeddings = []
for chunk in flattened_df['chunks']:
    embeddings.append(create_embeddings(chunk))
    
# Convertir a cadenas
flattened_df['embeddings'] = embeddings

# Resetear el índice
flattened_df = flattened_df.reset_index(drop=True)

# Insertar los embeddings en Chroma
for idx, row in flattened_df.iterrows():
    collection.add(
        ids=str(idx),  # Usar el índice como identificador único
        embeddings=row['embeddings'],
        metadatas={"chunk": row['chunks']}
    )
    