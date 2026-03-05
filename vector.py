from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
# EMBEDDINGS AND VECTOR STORE SETUP
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)
vector_store = Chroma(  
    collection_name="eng_data",
    persist_directory=db_location,
    embedding_function=embeddings
)


# STAGE DATA SECTION
stageDataFrame = pd.read_csv("vectorData/wrc_stages.csv")
unique_stages = stageDataFrame['stage_name'].unique()
stage_identity_docs = []

for stage in unique_stages:
    rows = stageDataFrame[stageDataFrame['stage_name'] == stage]
    row = rows.iloc[0]

    doc = Document(
        page_content=(
            f"Stage: {str(row['stage_name'])}\n"
            f"Country: {str(row['country'])}\n"
            f"Surface: {str(row['surface'])}\n"
            f"Distance: {float(row['distance_km'])} km\n"
            f"Average Speed: {float(row['avg_speed_kph'])} km/h\n"
            f"Corner Density: {float(row['corner_density'])}"
        ),
        metadata={
            "stage_name": str(row['stage_name']),
            "country": str(row['country']),
            "surface": str(row['surface']),
            "distance_km": float(row['distance_km']),
            "avg_speed_kph": float(row['avg_speed_kph']),
            "corner_density": float(row['corner_density']),
            "doc_type": "stage_identity"
        },
        id=f"stage-identity-{row['stage_name'].replace(' ', '_')}"
    )
    stage_identity_docs.append(doc)   
# CAR DATA SECTION
carDataFrame = pd.read_csv("vectorData/car_settings.csv")
# Clean car_name column
car_names = (
    carDataFrame["car_name"]
    .dropna()
    .map(str)
    .map(str.strip)
)
# Remove header-like and empty values
car_names = [
    name for name in car_names
    if name and name.lower() != "car_name"
]
unique_cars = sorted(set(car_names))
car_identity_docs = []
for car in unique_cars:
    car_rows = carDataFrame[carDataFrame["car_name"] == car]
    doc = Document(
        page_content=(
            f"Car Name: {car}\n"
            f"Class: {car_rows['class'].iloc[0] if 'class' in car_rows else 'Unknown'}"
        ),
        metadata={"car_name": car, "doc_type": "car_identity"},
        id=f"car-identity-{car.replace(' ', '_')}",
    )
    car_identity_docs.append(doc)

# add stage and car documents to vector store if not already added, this will create the vector store on disk and persist it for future use so we dont have to re add documents every time we run the code
if add_documents:
    documents = []
    ids = []
    # --- Stage documents ---
    for i, row in stageDataFrame.iterrows():
        document = Document(
            page_content=(
                f"Stage: {row['stage_name']}, "
                f"Surface: {row['surface']}, "
                f"Length: {row['distance_km']} km, "
                f"Average Speed: {row['avg_speed_kph']} km/h, "
                f"Corner Density: {row['corner_density']}"
            ),
            metadata={
                "stage_name": row["stage_name"],
                "country": row["country"],
                "surface": row["surface"],
                "distance_km": row["distance_km"],
                "avg_speed_kph": row["avg_speed_kph"]
            },
            id=f"stage-{i}"
        )
        ids.append(f"stage-{i}")
        documents.append(document)
    # --- Car documents ---
    for i, row in carDataFrame.iterrows():
        document = Document(
            page_content=(
                f"Car Name: {row['car_name']}, "
                f"Slider: {row['slider_name']}, "
                f"Minimum: {row['min_value']}, "
                f"Maximum: {row['max_value']}, "
                f"Has adjustment: {row['has_slider']}"
            ),
            metadata={
                "car_name": row['car_name'],
                "slider_name": row['slider_name'],
                "has_slider": row['has_slider'],
                "min_value": row['min_value'],
                "max_value": row['max_value']
            },
            id=f"car-{i}"  
        )
        ids.append(f"car-{i}")
        documents.append(document)
        
if add_documents:
    vector_store.add_documents(stage_identity_docs)
    vector_store.add_documents(car_identity_docs)
    vector_store.add_documents(documents=documents, ids=ids)
    
car_retriever = vector_store.as_retriever(
    search_kwargs={"k": 200, "filter": {"car_name": {"$contains": ""}}}
)

stage_retriever = vector_store.as_retriever(
    search_kwargs={"k": 200, "filter": {"stage_name": {"$contains": ""}}}
)
car_list_retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 100,
        "filter": {"doc_type": {"$in": ["car_identity"]}}
    }
)
stage_list_retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 100,
        "filter": {"doc_type": {"$in": ["stage_identity"]}}
    }
)