import os
import json
import csv
import re

import openai
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from SPARQLWrapper import SPARQLWrapper, JSON

# OpenAI API key
openai_api_key_file = open("../openai_api_key.txt", "r")
os.environ["OPENAI_API_KEY"] = openai_api_key_file.read().strip()
openai_api_key_file.close()

sparql_wd = SPARQLWrapper("https://query.wikidata.org/sparql")

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
for i, item in enumerate(data[66:68]):  # 66:71
    question = item['question_text']
    response = llm.predict(question)

    # qa_pairs[i] = dict()
    # qa_pairs[i]["question"] = question
    # qa_pairs[i]["response"] = response.strip()

    # Extract knowledge graph facts from the response
    llm_facts_json = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You will be provided with text, and your task is to extract RDF triples \
                and parse them into table format.\nSubject | Predicate | Object"
            },
            {
                "role": "user",
                "content": f"{response}"
            }
        ],
        temperature=0,
        max_tokens=256
    )

    llm_facts = llm_facts_json["choices"][0]["message"]["content"]

    # Using the extracted facts, generate some SPARQL queries
    sparql_json = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You will be provided with a table of RDF triples, and your task is to create a list of \
                           ASK SPARQL queries on Wikidata that can be used to verify those facts."
            },
            {
                "role": "user",
                "content": f"{llm_facts}"
            }
        ],
        temperature=0,
        max_tokens=2000
    )

    sparql_query_output = sparql_json["choices"][0]["message"]["content"]

    sparql_queries = re.findall(r"\bASK(?:\s+WHERE)?\b\s*\{[^}]*\}", sparql_query_output, re.IGNORECASE)

    print("Q:", question)
    print("A:", response.strip(), "\n")
    print("Facts:", llm_facts, "\n")
    print("SPARQL:", sparql_query_output, "\n")
    print("Queries:", sparql_queries, "\n")

    sparql_results = []

    for query in sparql_queries:
        sparql_wd.setQuery(query)
        sparql_wd.setReturnFormat(JSON)
        sparql_result = sparql_wd.query().convert()
        sparql_results.append(sparql_result["boolean"])
        print("Result:", sparql_result["boolean"], "\n")

    # qa_pairs[i]["llm_facts"] = llm_facts
    # qa_pairs[i]["sparql_queries"] = sparql_queries

    # Identify entities in the question
    entities_json = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You will be provided with text, and your task is to extract a numbered list of \
                           keywords from it."
            },
            {
                "role": "user",
                "content": f"{question}"
            }
        ],
        temperature=0.5,
        max_tokens=256
    )

    entities_output = entities_json["choices"][0]["message"]["content"]

    # Parse the list output into a Python list
    e_list = entities_output.split("\n")
    new_e_list = []

    print("Entities:", entities_output)

    # Remove the numbers from the list
    for k in range(len(e_list)):
        if e_list[k][:1].isdigit():
            idx = e_list[k].find(".")
            if idx != -1:
                new_e_list.append(e_list[k][idx+1:].lstrip())

    # Example SPARQL Query
    # SELECT ?subject ?predicate ?object
    # WHERE
    # {
    #     wd: Q1299 ?predicate ?object.
    # }

    print("Entities List:", new_e_list, "\n")

# Save the QA pairs in a JSON file
# with open("../output/qa_pairs.json", "w") as f:
#     json.dump(qa_pairs, f, indent=4)
