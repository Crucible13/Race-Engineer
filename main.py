from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import ollama
from vector import retreiver

#initialize ollama client
client = ollama.Client()
#define model 
model = OllamaLLM(model="Race-Engineer-Model-1")

template = """
You are an expert rally engineer with extensive experience in optimizing car setups for various rally stages. Your task is to provide detailed and tailored setup recommendations based on the specific characteristics of the stage, including terrain, weather conditions, and the driver's preferences.

Here is the response from the user: {user_input}

Here is the stage data (only for your reference based on which stage they say they are on to reccomend tweaks based on how the stage is): {stage_data}

Here is car information (for reference on what changes can be made and what limit the sliders have):  {car_data}

"""
#build prompt and response chain
prompt=ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n--------------------------------------")
    question = input("Enter input (q to quit): ")
    if question.lower() == "q":
        break
    stage_data = retreiver.invoke(question)
    result = chain.invoke({"user_input": question, "stage_data": stage_data, "car_data": []})
    print(result)
