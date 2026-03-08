# vector.py
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

import os
import pandas as pd

# -----------------------------
# EMBEDDINGS + VECTOR STORE
# -----------------------------
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

DB_LOCATION = "./chroma_langchain_db"
add_documents = not os.path.exists(DB_LOCATION)

vector_store = Chroma(
    collection_name="eng_data",
    persist_directory=DB_LOCATION,
    embedding_function=embeddings,
)

# -----------------------------
# MANUAL STAGE PROFILES 
# -----------------------------
manual_stage_profiles = {
    "Secto Finland": "fast, flowing, smooth",
    "Rally Pacifico": "mixed, low-grip",
    "Mediterraneo": "tarmac, abrasive, technical",
    "Monte Carlo": "technical, narrow, icy-risk",
    "Greece": "rough, rocky, technical",
    "Sweden": "fast, snow,"
}


def build_auto_stage_profile(row: pd.Series) -> str:
    """Derive an automatic stage profile from numeric + surface data."""
    tags = []

    avg_speed = float(row.get("avg_speed_kph", 0.0))
    corner_density = float(row.get("corner_density", 0.0))
    surface = str(row.get("surface", "")).lower()

    # Speed-based tags
    if avg_speed >= 110:
        tags.append("fast")
    elif avg_speed <= 80:
        tags.append("slow")
        tags.append("technical")
    else:
        tags.append("mixed")

    # Corner density tags
    if corner_density >= 0.45:
        if "technical" not in tags:
            tags.append("technical")
    elif corner_density <= 0.25:
        tags.append("flowing")

    # Surface tags
    if "gravel" in surface:
        tags.append("rough")
        tags.append("gravel")
    elif "tarmac" in surface:
        tags.append("tarmac")
    elif "snow" in surface:
        tags.append("snow")
        tags.append("low-grip")
    else:
        if surface:
            tags.append(surface)

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for t in tags:
        t = t.strip()
        if t and t not in seen:
            seen.add(t)
            deduped.append(t)

    return ", ".join(deduped)


def merge_auto_manual_profile(auto_profile: str, manual_profile: str | None) -> str:
    """Merge automatic and manual profiles into a single comma-separated string."""
    auto_tags = [t.strip() for t in auto_profile.split(",") if t.strip()] if auto_profile else []
    manual_tags = [t.strip() for t in manual_profile.split(",") if t.strip()] if manual_profile else []

    merged = []
    seen = set()

    for t in auto_tags + manual_tags:
        if t and t not in seen:
            seen.add(t)
            merged.append(t)

    return ", ".join(merged)


# -----------------------------
# TUNING DATA → DOCUMENTS
# -----------------------------
tuning_df = pd.read_csv("vectorData/tuning_suggestions.csv")

tuning_docs: list[Document] = []
for row in tuning_df.itertuples(index=False):
    doc = Document(
        page_content=(
            f"Slider: {row.slider}. "
            f"Range: {row.min_value} to {row.max_value}, step {row.step}. "
            f"Mechanical effect: {row.mechanical_effect}. "
            f"Driver feedback: {row.driver_feedback}. "
            f"Recommended adjustment: {row.recommended_adjustment}. "
            f"Stage conditions: {row.stage_conditions}."
        ),
        metadata={
            "doc_type": "tuning",
            "slider": row.slider,
            "min_value": row.min_value,
            "max_value": row.max_value,
            "step": row.step,
            "setup_area": getattr(row, "setup_area", None),
            "driver_feedback": row.driver_feedback,
            "stage_conditions": row.stage_conditions,
        },
    )
    tuning_docs.append(doc)

# -----------------------------
# STAGE DATA → IDENTITY DOCS
# -----------------------------
stage_df = pd.read_csv("vectorData/wrc_stages.csv")
unique_stages = stage_df["stage_name"].unique()

stage_identity_docs: list[Document] = []

for stage_name in unique_stages:
    rows = stage_df[stage_df["stage_name"] == stage_name]
    row = rows.iloc[0]

    auto_profile = build_auto_stage_profile(row)

    # Try to get rally name if present, else None
    rally_name = None
    if "rally" in stage_df.columns:
        rally_name = str(row.get("rally", "")).strip() or None

    manual_profile = manual_stage_profiles.get(rally_name, None) if rally_name else None
    merged_profile = merge_auto_manual_profile(auto_profile, manual_profile)

    doc = Document(
        page_content=(
            f"Stage: {str(row['stage_name'])}\n"
            f"Country: {str(row.get('country', 'Unknown'))}\n"
            f"Surface: {str(row['surface'])}\n"
            f"Distance: {float(row['distance_km'])} km\n"
            f"Average Speed: {float(row['avg_speed_kph'])} km/h\n"
            f"Corner Density: {float(row['corner_density'])}\n"
            f"Stage Profile: {merged_profile}"
        ),
        metadata={
            "stage_name": str(row["stage_name"]),
            "country": str(row.get("country", "Unknown")),
            "surface": str(row["surface"]),
            "distance_km": float(row["distance_km"]),
            "avg_speed_kph": float(row["avg_speed_kph"]),
            "corner_density": float(row["corner_density"]),
            "stage_profile": merged_profile,
            "rally": rally_name,
            "doc_type": "stage_identity",
        },
        id=f"stage-identity-{str(row['stage_name']).replace(' ', '_')}",
    )
    stage_identity_docs.append(doc)

# -----------------------------
# CAR DATA → IDENTITY DOCS
# -----------------------------
car_df = pd.read_csv("vectorData/car_settings.csv")

car_names = (
    car_df["car_name"]
    .dropna()
    .map(str)
    .map(str.strip)
)

car_names = [
    name for name in car_names
    if name and name.lower() != "car_name"
]

unique_cars = sorted(set(car_names))

car_identity_docs: list[Document] = []
for car in unique_cars:
    car_rows = car_df[car_df["car_name"] == car]
    car_class = (
        car_rows["class"].iloc[0]
        if "class" in car_rows.columns and not car_rows["class"].isna().all()
        else "Unknown"
    )

    doc = Document(
        page_content=(
            f"Car Name: {car}\n"
            f"Class: {car_class}"
        ),
        metadata={
            "car_name": car,
            "class": car_class,
            "doc_type": "car_identity",
        },
        id=f"car-identity-{car.replace(' ', '_')}",
    )
    car_identity_docs.append(doc)

# -----------------------------
# BULK STAGE + CAR DOCS
# -----------------------------
documents: list[Document] = []
ids: list[str] = []

if add_documents:
    # Stage documents (per row)
    for i, row in stage_df.iterrows():
        auto_profile = build_auto_stage_profile(row)

        rally_name = None
        if "rally" in stage_df.columns:
            rally_name = str(row.get("rally", "")).strip() or None

        manual_profile = manual_stage_profiles.get(rally_name, None) if rally_name else None
        merged_profile = merge_auto_manual_profile(auto_profile, manual_profile)

        document = Document(
            page_content=(
                f"Stage: {row['stage_name']}, "
                f"Surface: {row['surface']}, "
                f"Length: {row['distance_km']} km, "
                f"Average Speed: {row['avg_speed_kph']} km/h, "
                f"Corner Density: {row['corner_density']}, "
                f"Stage Profile: {merged_profile}"
            ),
            metadata={
                "stage_name": row["stage_name"],
                "country": row.get("country", "Unknown"),
                "surface": row["surface"],
                "distance_km": row["distance_km"],
                "avg_speed_kph": row["avg_speed_kph"],
                "corner_density": row["corner_density"],
                "stage_profile": merged_profile,
                "rally": rally_name,
            },
            id=f"stage-{i}",
        )
        ids.append(f"stage-{i}")
        documents.append(document)

    # Car documents (per slider row)
    for i, row in car_df.iterrows():
        document = Document(
            page_content=(
                f"Car Name: {row['car_name']}, "
                f"Slider: {row['slider_name']}, "
                f"Minimum: {row['min_value']}, "
                f"Maximum: {row['max_value']}, "
                f"Has adjustment: {row['has_slider']}"
            ),
            metadata={
                "car_name": row["car_name"],
                "slider_name": row["slider_name"],
                "has_slider": row["has_slider"],
                "min_value": row["min_value"],
                "max_value": row["max_value"],
            },
            id=f"car-{i}",
        )
        ids.append(f"car-{i}")
        documents.append(document)

# -----------------------------
# ADD DOCUMENTS TO VECTOR STORE
# -----------------------------
if add_documents:
    vector_store.add_documents(tuning_docs)
    vector_store.add_documents(stage_identity_docs)
    vector_store.add_documents(car_identity_docs)
    vector_store.add_documents(documents=documents, ids=ids)

# -----------------------------
# RETRIEVERS
# -----------------------------
car_retriever = vector_store.as_retriever(
    search_kwargs={"k": 200, "filter": {"car_name": {"$contains": ""}}}
)

stage_retriever = vector_store.as_retriever(
    search_kwargs={"k": 200, "filter": {"stage_name": {"$contains": ""}}}
)

car_list_retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 100,
        "filter": {"doc_type": {"$in": ["car_identity"]}},
    }
)

stage_list_retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 100,
        "filter": {"doc_type": {"$in": ["stage_identity"]}},
    }
)

tuning_retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 20,
        "filter": {"doc_type": "tuning"},
    }
)

# -----------------------------
# SETUP-FOCUSED TUNING RETRIEVAL
# -----------------------------
def retrieve_tuning_for_setup(
    symptom: str,
    surface: str | None = None,
    weather: str | None = None,
    stage_profile: str | None = None,
):
    """
    Retrieve tuning suggestions for a setup-focused question.

    symptom: driver feedback text (e.g. 'pushes mid-corner')
    surface: gravel / tarmac / snow / mixed
    weather: dry / damp / wet / flooded
    stage_profile: merged profile string (e.g. 'fast, flowing, rough')
    """
    query_parts = [f"Driver feedback: {symptom}"]

    if surface:
        query_parts.append(f"Surface: {surface}")
    if weather:
        query_parts.append(f"Weather: {weather}")
    if stage_profile:
        query_parts.append(f"Stage profile: {stage_profile}")

    query = " | ".join(query_parts)

    return tuning_retriever.invoke(query)