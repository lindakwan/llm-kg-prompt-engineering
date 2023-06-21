import os
import json
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# Load the data
file = open("../data/nq_data_simple/train25.json", "r")
data = json.load(file)

# Create the language model
llm = OpenAI(temperature=0)

# Create a list of QA pairs
qa_pairs = dict()

# Generate a response for each question
for i, item in enumerate(data):
    question = item['question_text']
    response = llm.predict(question)

    qa_pairs[i] = dict()
    qa_pairs[i]["question"] = question
    qa_pairs[i]["response"] = response.strip()

    # Generate a SPARQL query for each question
    prompt = PromptTemplate(
        input_variables=["question"],
        template="Generate a SPARQL query to extract facts from Wikidata related to the following question: {question}"
    )

    sparql_query = llm(prompt.format(question=question))

    print("Q:", question)
    print("A:", response.strip(), "\n")
    print("SPARQL:", sparql_query, "\n")

    qa_pairs[i]["sparql_query"] = sparql_query

# Save the QA pairs in a JSON file
with open("../output/qa_pairs.json", "w") as f:
    json.dump(qa_pairs, f, indent=4)
