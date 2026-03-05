from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import ollama
from rapidfuzz import process
from vector import (
    stage_retriever,
    car_retriever,
    car_list_retriever,
    stage_list_retriever,
)

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

STRICT RULES FOR GENERAL QUESTIONS
• Answer ONLY using information found in {eng_data}.
• Summaries must be grounded in the retrieved documents.
• Do not add external knowledge.

OUTPUT STYLE
• Be direct, technical, and concise.
• No filler, no speculation, no invented content.
• No extra tuning details unless the user explicitly asks for them.
• When listing items, output clean lists with no commentary unless requested.

USER REQUEST
{user_input}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


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


print("Welcome back. I’m your Race Engineer, here to give you clear, actionable information for the stage ahead.")

while True:
    print("\n--------------------------------------")
    question = input("Enter input (q to quit): ")
    if question.lower() == "q":
        break

    lower_q = question.lower().strip()

    # 1. Rule-based intent detection first
    if any(phrase in lower_q for phrase in [
        "list all cars", "list cars", "all cars", "show all cars",
        "give all cars", "what cars", "cars?"
    ]):
        intent = "car listing"

    elif any(phrase in lower_q for phrase in [
        "list all stages", "list stages", "all stages", "show all stages",
        "give all stages", "what stages", "stages?"
    ]):
        intent = "stage listing"

    elif "setup" in lower_q or "tune" in lower_q or "adjust" in lower_q:
        intent = "setup"

    else:
        intent = detect_intent(question).strip().lower()

    # 2. Select retriever based on intent
    detected_car = None
    detected_stage = None

    if intent == "setup":
        # For setup, we want both car + stage, so we fuzzy-detect them
        detected_car, detected_stage = fuzzy_detect_car_and_stage(question)

        # For stage-aware setup, we primarily retrieve stage docs
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

    eng_docs = retriever.invoke(question)

    # 3. Fast path: car listing
    if "car" in intent and "listing" in intent:
        car_names = sorted([doc.metadata["car_name"] for doc in eng_docs])
        numbered = "\n".join(f"{i+1}. {name}" for i, name in enumerate(car_names))
        print(numbered)
        continue

    # 4. Fast path: stage listing
    if "stage" in intent and "listing" in intent:
        stage_names = sorted([doc.metadata["stage_name"] for doc in eng_docs])
        numbered = "\n".join(f"{i+1}. {name}" for i, name in enumerate(stage_names))
        print(numbered)
        continue

    # 5. For setup, enrich eng_docs with car docs as well
    if intent == "setup":
        detected_car, detected_stage = fuzzy_detect_car_and_stage(question)

    all_docs = []

    # Always retrieve car slider docs if a car was detected
    if detected_car:
        car_docs = car_retriever.invoke(detected_car)
        all_docs.extend(car_docs)

    # Always retrieve stage metadata docs if a stage was detected
    if detected_stage:
        stage_docs = stage_retriever.invoke(detected_stage)
        all_docs.extend(stage_docs)

    # If either car or stage is missing, the LLM will ask for only the missing piece
    eng_docs = all_docs

    # Build enriched user input
    setup_context = question
    if detected_car:
        setup_context += f"\n\nDetected car: {detected_car}"
    if detected_stage:
        setup_context += f"\nDetected stage: {detected_stage}"


        eng_text = "\n\n".join([doc.page_content for doc in eng_docs])
        result = chain.invoke({"user_input": setup_context, "eng_data": eng_text})
        print(result)
        continue

    # 6. LLM path — general reasoning
    eng_text = "\n\n".join([doc.page_content for doc in eng_docs])
    result = chain.invoke({"user_input": question, "eng_data": eng_text})
    print(result)