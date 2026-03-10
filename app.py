from flask import Flask, render_template, request, jsonify, stream_with_context, Response
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import ollama
from rapidfuzz import process
from vector import (
    stage_retriever,
    car_retriever,
    car_list_retriever,
    stage_list_retriever,
    retrieve_tuning_for_setup
)

#define app
app = Flask(__name__)
# initialize ollama client
client = ollama.Client()

# define model
model = OllamaLLM(model="Race-Engineer-Model-1")

template = """
You are a Rally Engineer for EA SPORTS WRC. Use ONLY {eng_data}. Never invent cars, stages, sliders, values, or systems.

Think internally and silently. Never reveal chain‑of‑thought. Output only the final answer.

SETUPS:
- Require car + stage; if one missing, ask only for the missing one.
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
Handbrake 1139.3–2827.3 (25)
F Drive Lock 0–37% (0.5%)
F Brake Lock 0–37% (0.5%)
F Preload 0–96.25 (1.5)
R Drive Lock 0–44% (0.5%)
R Brake Lock 0–42% (0.5%)
R Preload 0–100 (2.0)
Gears 0.200–1.200 (0.02)
Final Drive 0.100–0.300 (0.005)
Slow/Fast Bump −5.00–+5.00 (0.20)
Bump Div 0.00–1.30 (0.2)
Ride Height 30–70 (1)
Spring Rate 20–100 (1.5)
ARB 0–66 (1)

OUTPUT FORMAT RULES (MANDATORY)
You must output the setup in a format that the frontend parser can read.

1. SECTION HEADERS
   • A section header is a single line containing only the section name.
   • Example:
       Tyres
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

Your output must ONLY contain:
   - Section headers
   - Bullet rows with item/value pairs

USER REQUEST
{user_input}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

########## helper functions #########
def detect_intent(query: str) -> str:
    """LLM-based intent classifier used only when rule-based logic doesn't decide."""
    cls_prompt = f"""
    Classify the user's intent using ONLY these labels:
    - car
    - stage
    - car listing
    - stage listing
    - other

    Classification rules:
    • If the user asks to "list", "list all cars", "show all", "give all", "what cars", "all cars", classify as 'car listing'.
    • If the user asks to "list", "show all", "give all", "what stages", "all stages", classify as 'stage listing'.
    • If the user mentions a specific car name, classify as 'car'.
    • If the user mentions a specific stage name, classify as 'stage'.
    • Otherwise classify as 'other'.

    User query:
    "{query}"
    """
    return model.invoke(cls_prompt).strip().lower()


def fuzzy_detect_car_and_stage(user_text: str):
    """Use rapidfuzz to detect car_name and stage_name from the user text."""
    lower_q = user_text.lower()

    # get all car identity docs
    car_docs = car_list_retriever.invoke("")
    car_names = [d.metadata["car_name"] for d in car_docs]

    # get all stage identity docs
    stage_docs = stage_list_retriever.invoke("")
    stage_names = [d.metadata["stage_name"] for d in stage_docs]

    detected_car = None
    detected_stage = None

    if car_names:
        car_match = process.extractOne(
            lower_q,
            car_names,
        )
        if car_match and car_match[1] >= 55:
            detected_car = car_match[0]

    if stage_names:
        stage_match = process.extractOne(
            lower_q,
            stage_names,
        )
        if stage_match and stage_match[1] >= 55:
            detected_stage = stage_match[0]

    return detected_car, detected_stage
########## helper functions #########

######## APP ROUTES ########
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")
    lower_in = user_input.lower().strip()

    # 1. Rule-based intent detection
    if any(phrase in lower_in for phrase in [
        "list all cars", "list cars", "all cars", "show all cars",
        "give all cars", "what cars", "cars?"
    ]):
        intent = "car listing"

    elif any(phrase in lower_in for phrase in [
        "list all stages", "list stages", "all stages", "show all stages",
        "give all stages", "what stages", "stages?"
    ]):
        intent = "stage listing"

    elif "setup" in lower_in or "tune" in lower_in or "adjust" in lower_in:
        intent = "setup"

    else:
        intent = detect_intent(user_input).strip().lower()

    # 2. Select retriever
    detected_car = None
    detected_stage = None

    if intent == "setup":
        detected_car, detected_stage = fuzzy_detect_car_and_stage(user_input)
        retriever = stage_retriever

    elif intent == "car":
        retriever = car_retriever

    elif intent == "stage":
        retriever = stage_retriever

    elif intent == "car listing":
        retriever = car_list_retriever

    elif intent == "stage listing":
        retriever = stage_list_retriever

    else:
        retriever = car_retriever

    eng_docs = retriever.invoke(user_input)

    # 3. Fast paths
    if intent == "car listing":
        car_names = sorted([doc.metadata["car_name"] for doc in eng_docs])
        numbered = "\n".join(f"{i+1}. {name}" for i, name in enumerate(car_names))
        return jsonify({"response": numbered})

    if intent == "stage listing":
        stage_names = sorted([doc.metadata["stage_name"] for doc in eng_docs])
        numbered = "\n".join(f"{i+1}. {name}" for i, name in enumerate(stage_names))
        return jsonify({"response": numbered})

    # 4. Setup enrichment
    if intent == "setup":
        detected_car, detected_stage = fuzzy_detect_car_and_stage(user_input)

    all_docs = []

    if detected_car:
        all_docs.extend(car_retriever.invoke(detected_car))

    if detected_stage:
        all_docs.extend(stage_retriever.invoke(detected_stage))

    eng_docs = all_docs
    if intent == "setup":
        detected_car, detected_stage = fuzzy_detect_car_and_stage(user_input)

    all_docs = []

    if detected_car:
        all_docs.extend(car_retriever.invoke(detected_car))

    if detected_stage:
        stage_docs = stage_retriever.invoke(detected_stage)
        all_docs.extend(stage_docs)

        # Extract stage profile from metadata
        stage_profile = None
        for d in stage_docs:
            if "stage_profile" in d.metadata:
                stage_profile = d.metadata["stage_profile"]
                break
    else:
        stage_profile = None

    # Retrieve tuning docs
    issues = user_input
    surface = None
    weather = None

    # Try to extract surface/weather from user text
    for s in ["gravel", "tarmac", "snow", "mixed"]:
        if s in user_input.lower():
            surface = s

    for w in ["dry", "damp", "wet", "flooded"]:
        if w in user_input.lower():
            weather = w

    tuning_docs = retrieve_tuning_for_setup(
        symptom=issues,
        surface=surface,
        weather=weather,
        stage_profile=stage_profile
    )

    all_docs.extend(tuning_docs)

    eng_docs = all_docs

    setup_context = user_input
    if detected_car:
        setup_context += f"\n\nDetected car: {detected_car}"
    if detected_stage:
        setup_context += f"\nDetected stage: {detected_stage}"

    eng_text = "\n\n".join([doc.page_content for doc in eng_docs])

    # 5. STREAMING RESPONSE
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

@app.route("/cars")
def get_cars():
    docs = car_list_retriever.invoke("")
    cars = sorted([d.metadata["car_name"] for d in docs])
    return jsonify({"cars": cars})

@app.route("/")
def mainScreen():
    return render_template("index.html")

@app.route("/setupCreation")
def setupCreation():
    return render_template("setupCreation.html")

if __name__ == "__main__":
    app.run(debug=True)
