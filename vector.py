# vector.py
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

import os
import pandas as pd

# -----------------------------
# EMBEDDINGS + VECTOR STORES
# -----------------------------
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

DB_LOCATION = "./chroma_langchain_db"
add_documents = not os.path.exists(DB_LOCATION)

car_identity_store = Chroma(
    collection_name="car_identity",
    persist_directory=DB_LOCATION,
    embedding_function=embeddings,
)

stage_identity_store = Chroma(
    collection_name="stage_identity",
    persist_directory=DB_LOCATION,
    embedding_function=embeddings,
)

tuning_store = Chroma(
    collection_name="tuning_docs",
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
    "Sweden": "fast, snow,",
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


def add_in_batches(store: Chroma, docs: list[Document], batch_size: int = 5000):
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        print(f"Adding batch {i // batch_size + 1} to {store._collection.name} ({len(batch)} docs)...")
        store.add_documents(batch)


# -----------------------------
# TUNING DATA → TUNING DOCS
# -----------------------------
tuning_docs: list[Document] = []

tuning_df = pd.read_csv("vectorData/tuning_suggestions.csv")
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

setup_df = pd.read_csv("vectorData/wrc_setups.csv")
for row in setup_df.itertuples(index=False):
    doc = Document(
        page_content=(
            f"Car: {row.Car}. "
            f"Class: {row.Class}. "
            f"Rally: {row.Rally}. "
            f"Stage: {row.Stage}. "
            f"Section: {row.Section}. "
            f"Adjustment: {row.Adjustment}. "
            f"Value: {row.Value}."
        ),
        metadata={
            "doc_type": "tuning_setup",
            "car": row.Car,
            "car_class": row.Class,
            "rally": row.Rally,
            "stage": row.Stage,
            "section": row.Section,
            "adjustment": row.Adjustment,
            "value": row.Value,
        },
    )
    tuning_docs.append(doc)

# -----------------------------
# STAGE DATA → STAGE IDENTITY DOCS
# -----------------------------
stage_df = pd.read_csv("vectorData/wrc_stages.csv")
unique_stages = stage_df["stage_name"].unique()

stage_identity_docs: list[Document] = []

for stage_name in unique_stages:
    rows = stage_df[stage_df["stage_name"] == stage_name]
    row = rows.iloc[0]

    auto_profile = build_auto_stage_profile(row)

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
# CAR DATA → CAR IDENTITY DOCS
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
# PERSIST DOCUMENTS (ONE-TIME)
# -----------------------------
if add_documents:
    print("Populating Chroma collections...")

    if car_identity_docs:
        add_in_batches(car_identity_store, car_identity_docs, batch_size=5000)

    if stage_identity_docs:
        add_in_batches(stage_identity_store, stage_identity_docs, batch_size=5000)

    if tuning_docs:
        add_in_batches(tuning_store, tuning_docs, batch_size=5000)

    print("Chroma collections populated.")

car_identity_retriever = car_identity_store.as_retriever(search_kwargs={"k": 1})
stage_identity_retriever = stage_identity_store.as_retriever(search_kwargs={"k": 1})
car_list_retriever = car_identity_store.as_retriever(search_kwargs={"k": 100})
stage_list_retriever = stage_identity_store.as_retriever(search_kwargs={"k": 100})
tuning_retriever = tuning_store.as_retriever(search_kwargs={"k": 3})
__all__ = [
    "car_identity_store",
    "stage_identity_store",
    "tuning_store",
    "manual_stage_profiles",
    "build_auto_stage_profile",
    "merge_auto_manual_profile",
]