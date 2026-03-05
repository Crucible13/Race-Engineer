from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

stageDataFrame = pd.read_csv("stageData/wrc_stages.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []
    for i, row in stageDataFrame.iterrows():
        document = Document(
            page_content=f"Stage: {row['stage_name']}, , Surface: {row['surface']}, Length: {row['distance_km']} km, Average Speed: {row['avg_speed_kph']} km/h, Corner Density: {row['corner_density']}",
            metadata={"stage_name": row["stage_name"], "country": row["country"], "surface": row["surface"], "distance_km": row["distance_km"], "avg_speed_kph": row["avg_speed_kph"]},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)
        
vector_store = Chroma(  
    collection_name="stage_data",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    
retreiver = vector_store.as_retriever(
    search_kwargs={"k": 10}
    )