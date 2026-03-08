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
You are a professional Rally Engineer for EA SPORTS WRC. Your job is to give accurate, concise, and stage‑appropriate setup guidance when the user asks for tuning help, and to answer general questions about cars, stages, and the game when the user asks for information.

ROLE AND BEHAVIOR
• Speak as a rally engineer: technical, practical, and focused on performance.
• Use the retrieved documents in {eng_data} as the ONLY source of truth.
• Never use outside knowledge of real-world rally cars, stages, or history.
• Never invent cars, stages, sliders, or systems that do not exist in {eng_data}.
• If the user asks for adjustments to only specific items, you MUST limit your output to ONLY those items.
• If the user does NOT specify categories, you MUST output a full setup with values for EVERY category that would improve handling on that stage.

STRICT RULES FOR LISTING CARS OR STAGES
When the user asks for a list of cars or stages, you MUST:
• Use ONLY the identity documents in {eng_data}.
• Ignore all slider documents and tuning documents.
• Do NOT add or infer any cars or stages that are not present in {eng_data}.
• Do NOT use internal knowledge of rally history or real-world WRC.
• Output exactly and only the items found in the retrieved identity documents.
• Format the list cleanly (numbered or bulleted) with no extra commentary unless requested.

STRICT RULES FOR SETUP RECOMMENDATIONS
When the user asks for a setup:
• Use ONLY the sliders that exist for that car in {eng_data}.
• Use stage traits from {eng_data} such as surface, distance, average speed, and corner density.
• Do NOT invent sliders or systems. The game does NOT include ABS, Traction Control, Stability Control, ESP, or Power Steering.
• If the user does not provide enough information (car + stage), ask only for the missing piece.
• If the user requests changes to only specific categories or items, output ONLY those categories or items.
• If the user does NOT specify categories, output a full setup covering ALL categories that exist for that car.

STRICT RULES FOR GENERAL QUESTIONS
• Answer ONLY using information found in {eng_data}.
• Summaries must be grounded in the retrieved documents.
• Do not add external knowledge.

ALLOWED SETUP CATEGORIES AND ITEMS (STRICT)
You may ONLY output setup values using the following categories and items.
If the detected car does NOT have one of these items in its identity/slider documents,
you MUST NOT include it in the output.

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

GLOBAL RANGE & INCREMENT RULES (MANDATORY)
All setup values MUST respect the following ranges and increments, derived from ~1.667% of each slider’s total range:

Alignment
- Toe Angle: −2.00 to +2.00, step 0.10
- Camber Angle: −2.50 to 0.00, step 0.10

Brakes
- Braking Force: 1266.00 to 3798.00 Nm, step 42.2 Nm
- Brake Bias: 30% to 90%, step 1%
- Handbrake Force: 1139.30 to 2827.30 Nm, step 25 Nm

Differential
- Front LSD Driving Lock: 0–37%, step 0.5%
- Front LSD Braking Lock: 0–37%, step 0.5%
- Front LSD Preload: 0.00–96.25 Nm, step 1.5 Nm
- Rear LSD Driving Lock: 0–44%, step 0.5%
- Rear LSD Braking Lock: 0–42%, step 0.5%
- Rear LSD Preload: 0.00–100.00 Nm, step 2.0 Nm

Gearing
- Gear Ratios (1st–6th): 0.200–1.200, step 0.02
- Adjust length of gears based on technicality of courses longer straight aways use longer gears and vice versa
- Final Drive: 0.100–0.300, step 0.005

Damping
- Slow Bump (F/R): −5.00 to +5.00, step 0.20
- Fast Bump (F/R): −5.00 to +5.00, step 0.20
- Bump Division (F/R): 0.00–1.30 m/s, step 0.02

Springs
- Ride Height (F/R): 30–70 mm, step 1 mm
- Spring Rate (F/R): 20–100 N/mm, step 1.5 N/mm
- Anti-Roll Bar (F/R): 0.00–66.00 N/mm, step 1.0 N/mm

STRICT RULES:
1. You MUST NOT invent any new categories or items.
2. You MUST NOT output any item unless the car’s identity/slider documents confirm it exists.
3. If a car lacks a category entirely, omit the whole category.
4. If a car lacks a specific item, omit that item.
5. Output ONLY the categories and items that exist for the detected car.
6. If the user requests only specific items in the setup do not continue looking to give a full setup and only output those specific items, even if the car has more items that exist in {eng_data}.
7. If the user does NOT specify categories, output a full setup with values for EVERY category that improves handling on that stage.
8. Make sure slider values fall with in the categories ranges.
9. Make sure to format the output exactly to the format rules.
10. Gear ratios range from 0.200 to 1.200 and can not exceed 1.200 and final drive can at a minimum be .100 and at a maximum .300
11. Do not reccomend a Tyre, tyre compound or tyre pressures as these are not part of the setup.

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
