from flask import Flask, render_template, request, jsonify, stream_with_context, Response
from langchain_ollama.llms import OllamaLLM
import ollama

from vector import (
    stage_identity_retriever,
    car_identity_retriever,
    tuning_store,
    car_list_retriever,
)

# -----------------------------
# Flask + Ollama Setup
# -----------------------------
app = Flask(__name__)
client = ollama.Client()
model = OllamaLLM(model="Race-Engineer-Model-1")
template = """
You are a Rally Engineer for EA SPORTS WRC. Use ONLY {eng_data}. Never invent cars, stages, sliders, values, or systems.

Think internally and silently. Never reveal chain‑of‑thought. Output only the final answer. dont tell what each change does just the adjustment and its value

SETUPS:
- Require car + stage; if stage missing, make it broad for the rally selected.
- Use ONLY sliders that exist for that car in {eng_data}.
- Use ONLY stage traits from {eng_data}.
- If user requests specific items, output ONLY those items.
- If user does not specify categories, output a full setup for all categories that exist for that car.
- Respect all global ranges and increments exactly.
- Do NOT include tyres, compounds, or pressures.
- Do NOT output any category or item not present for that car.

LISTING:
- Use ONLY identity docs.
- No inference. No additions. Only what exists in {eng_data}.
- Output clean bullet lists only.

FORMAT:
- Only section headers + bullet rows.
- Bullet rows use "-", "•", or "*".
- Item/value pairs use ":", "=", or "—".
- No tables, pipes, paragraphs, or numbered lists.
- Never use em dashes (—). Use a normal hyphen (-) instead.

You must NEVER include tyres, brake pad compounds, brake fluid, steering ratio, or global ranges in the final setup. These items are NOT adjustable sliders and must be ignored even if provided by the user.

CATEGORIES AND ITEMS:

Alignment
- (Front) Toe Angle
- (Front) Camber Angle
- (Rear) Toe Angle
- (Rear) Camber Angle

Brakes
- Braking Force
- Brake Bias
- Handbrake Force

Differential
- (Front) LSD Driving Lock
- (Front) LSD Braking Lock
- (Front) LSD Preload
- (Rear) LSD Driving Lock
- (Rear) LSD Braking Lock
- (Rear) LSD Preload

Gearing
- 1st Gear
- 2nd Gear
- 3rd Gear
- 4th Gear
- 5th Gear
- 6th Gear (only if present in {eng_data})
- Final Drive

Damping
- (Front) Slow Bump
- (Front) Fast Bump
- (Front) Bump Division
- (Front) Slow Rebound
- (Rear) Slow Bump
- (Rear) Fast Bump
- (Rear) Bump Division
- (Rear) Slow Rebound

Springs
- (Front) Ride Height
- (Front) Spring Rate
- (Front) Anti-Roll Bar
- (Rear) Ride Height
- (Rear) Spring Rate
- (Rear) Anti-Roll Bar

GLOBAL RANGES:
Toe −2.00–+2.00 (1.00)
Camber −2.50–0.00 (0.982)
Braking Force 1266–3798 (42.2)
Brake Bias 30–90% (1%)
Handbrake Force 1139.3–2827.3 (25)
F Drive Lock 0–37% (0.5%)
F Brake Lock 0–37% (0.5%)
F Preload 0–96.25 (1.5)
R Drive Lock 0–44% (0.5%)
R Brake Lock 0–42% (0.5%)
R Preload 0–100 (2.0)
Gears 0.200–1.200 (0.02)
Final Drive 0.100–0.300 (0.005)
Slow/Fast Bump −5.00–+5.00 (0.20)
Bump Divison 0.00–1.30 (0.2)
Ride Height 30–70 (1)
Spring Rate 20–100 (1.5)
ARB 0–66 (1)

You must NEVER output global ranges in the final setup. They are only constraints for generating valid values.
OUTPUT FORMAT RULES (MANDATORY)
You must output the setup in a format that the frontend parser can read.

1. SECTION HEADERS
   • A section header is a single line containing only the section name.
   • Example:
       Alignment
       Brakes
       Suspension

2. SETUP ROWS
   • Each setup row MUST be a bullet point starting with "-", "•", or "*".
   • Each row MUST contain an item and a value.
   • The item and value MUST be separated using one of the following:
       - Colon (Item: Value)
       - Equals (Item = Value)
       - Em dash (Item — Value)


3. STRICT PROHIBITIONS
   • Do NOT output tables.
   • Do NOT output pipe characters "|".
   • Do NOT output multi-column formatting.
   • Do NOT output paragraphs.
   • Do NOT output numbered lists.
   • Do NOT output anything except section headers and bullet rows.
   
- If the user mentions any slider, system, or value not in this list or not present for that car in {eng_data}, you must ignore it completely and never approximate, rename, or replace it.

Your output must ONLY contain: (STRICTLY NO EXCEPTIONS)
   - Section headers
   - Bullet rows with item/value pairs
   
ignore everything not in {eng_data}. Never attempt to fill in missing values or invent items. If the user requests something not in {eng_data}, respond that you don't have that information.
USER REQUEST
{user_input}
"""

# -----------------------------
# Tuning Retrieval
# -----------------------------
def retrieve_tuning_for_setup(car=None, surface=None, weather=None, stage_profile=None):
    """Retrieve tuning documents from tuning_store using clean, deterministic keywords."""
    query_parts = []

    if car:
        query_parts.append(str(car))

    if surface:
        query_parts.append(str(surface))

    if weather:
        query_parts.append(str(weather))

    if stage_profile:
        query_parts.append(str(stage_profile))

    # Build clean query
    query = " ".join(query_parts).strip()
    if not query:
        return []

    # Retrieve only the top 3 most relevant tuning rows
    retriever = tuning_store.as_retriever(search_kwargs={"k": 3})
    return retriever.invoke(query)


# -----------------------------
# SETUP-ONLY CHAT ENDPOINT
# -----------------------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()

    detected_car = data.get("Car")
    detected_stage = data.get("Stage")
    detected_rally = data.get("Rally")
    issues = data.get("Issues")
    surface = data.get("Surface")
    weather = data.get("Weather")

    # -----------------------------
    # Retrieve identity + tuning docs
    # -----------------------------
    car_docs = car_identity_retriever.invoke(detected_car)
    stage_docs = stage_identity_retriever.invoke(detected_stage)

    stage_profile = None
    for d in stage_docs:
        if "stage_profile" in d.metadata:
            stage_profile = d.metadata["stage_profile"]
            break

    tuning_docs = retrieve_tuning_for_setup(
        car=detected_car,
        surface=surface,
        weather=weather,
        stage_profile=stage_profile
    )

    all_docs = car_docs + stage_docs + tuning_docs

    # -----------------------------
    # Build ENG DATA
    # -----------------------------
    eng_text = "\n\n".join(doc.page_content for doc in all_docs)
    print("=== ENG DATA ===")
    print(eng_text)
    print("================")

    # -----------------------------
    # Build setup context
    # -----------------------------
    setup_context = (
        f"Car: {detected_car}\n"
        f"Stage: {detected_stage}\n"
        f"Rally: {detected_rally}\n"
        f"Surface: {surface}\n"
        f"Weather: {weather}\n"
        f"Issues: {issues}"
    )

    # -----------------------------
    # STREAMING RESPONSE
    # -----------------------------
    def generate():
        result = client.chat(
            model="Race-Engineer-Model-1",
            messages=[{
                "role": "user",
                "content": template.format(
                    user_input=setup_context,
                    eng_data=eng_text
                )
            }],
            stream=True
        )

        for chunk in result:
            if "message" in chunk and "content" in chunk["message"]:
                yield chunk["message"]["content"]

    return Response(stream_with_context(generate()), mimetype="text/plain")


#Routes
@app.route("/")
def mainScreen():
    return render_template("index.html")

@app.route("/setupCreation")
def setupCreation():
    return render_template("setupCreation.html")
@app.route("/cars")
def get_cars():
    docs = car_list_retriever.invoke("")  
    cars = sorted([d.metadata["car_name"] for d in docs])
    return jsonify({"cars": cars})
    


if __name__ == "__main__":
    app.run(debug=True)