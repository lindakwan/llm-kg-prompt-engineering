import os
import json
import csv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# Load the data
# file = open("../data/nq_data_simple/train25.json", "r")
# data = json.load(file)

# Load the data
file = open("../data/mmlu_test/high_school_geography_test.csv", "r")
csv_reader = csv.reader(file, delimiter=',')
data = []
for row in csv_reader:
    data.append({"question_text": row[0], "choices": row[1:-1], "correct_answer": row[-1]})

# Create the language model
llm = OpenAI(temperature=0)

# Create a list of QA pairs
# qa_pairs = dict()

# Generate a response for each question
for i, item in enumerate(data[:5]):
    question = item['question_text']
    response = llm.predict(question)

    # qa_pairs[i] = dict()
    # qa_pairs[i]["question"] = question
    # qa_pairs[i]["response"] = response.strip()

    # Extract knowledge graph facts from the response
    llm_facts_prompt = PromptTemplate(
        input_variables=["response"],
        template="Extract knowledge graph facts from the following: {response}"
    )

    llm_facts = llm_facts_prompt.format(response=response)

    # Using the extracted facts, generate some SPARQL queries
    sparql_prompt = PromptTemplate(
        input_variables=["facts"],
        template="Convert the following facts to SPARQL queries: {facts}"
    )

    sparql_queries = llm(sparql_prompt.format(facts=llm_facts))

    # Generate a SPARQL query for each question
    # prompt = PromptTemplate(
    #     input_variables=["question"],
    #     template="Generate a SPARQL query to extract Wikidata facts for the following question: {question}"
    # )

    # sparql_query = llm(prompt.format(question=question))

    print("Q:", question)
    print("A:", response.strip(), "\n")
    print("Facts:", llm_facts, "\n")
    print("SPARQL:", sparql_queries, "\n")

    # print("SPARQL:", sparql_query, "\n")

    # qa_pairs[i]["llm_facts"] = llm_facts
    # qa_pairs[i]["sparql_queries"] = sparql_queries

# Save the QA pairs in a JSON file
# with open("../output/qa_pairs.json", "w") as f:
#     json.dump(qa_pairs, f, indent=4)
