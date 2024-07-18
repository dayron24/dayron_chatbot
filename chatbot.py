import os
from openai import AzureOpenAI
import dotenv
import chromadb

dotenv.load_dotenv()

client = AzureOpenAI(
    azure_endpoint="https://hatchworksai.openai.azure.com/",
    api_key=os.environ['AZURE_OPENAI_KEY'],
    api_version="2023-10-01-preview", 
)
deployment = "gpt-35"

def create_embeddings(text, model="ada-02"):
    # Create embeddings for each document chunk
    embeddings = client.embeddings.create(input = text, model=model).data[0].embedding
    return embeddings

chroma_client = chromadb.PersistentClient(path="./DB/")

# Nombre de la colección
collection_name = "embeddings_collection"

collection = chroma_client.get_collection(name=collection_name)

#user_input = "Where are Dayron working?"

def chatbot(user_input):
    # Convert the question to a query vector
    query_vector = create_embeddings(user_input)

    # Buscar los documentos más similares en Chroma
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=8  # Número de vecinos más cercanos a recuperar
    )
    # Añadir los documentos a la consulta para proporcionar contexto
    history = []
    for metadata in results["metadatas"][0]:
        history.append(metadata["chunk"])

    # Combinar la historia y la entrada del usuario
    history.append(user_input)
    #print(history[0:-1])
    # create a message object
    messages=[
        #TODO: Meterle el contexto/
        {"role": "system", "content": f"You are an AI assiatant that helps with AI questions, mostly about information from a person named Dayron, use this information to answer the question: {history[0:-1]}. don't say 'Based on the information provided or somethin like that'"},
        {"role": "user", "content": history[-1]}
    ]

    # use chat completion to generate a response
    response = client.chat.completions.create(
        model="gpt-35",
        temperature=0.4,
        max_tokens=800,
        messages=messages
    )

    return response.choices[0].message.content

# Bucle para interactuar continuamente con el usuario
while True:

    user_input = input("Enter your question (or 'exit' to quit): ")

    if user_input.lower() == 'exit':
        print("Goodbye!")
        break

    response = chatbot(user_input)
    print(response)
    