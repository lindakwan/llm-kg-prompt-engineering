import os
import json
import csv
import re
import subprocess
import ast

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
for i, item in enumerate(data[66:67]):  # 66:71
    question = item['question_text']
    response = llm.predict(question)

    print("Q:", question)
    print("A:", response.strip(), "\n")

    # qa_pairs[i] = dict()
    # qa_pairs[i]["question"] = question
    # qa_pairs[i]["response"] = response.strip()

    question_escaped = question.replace('"', '\\\"')
    response_escaped = response.strip().replace('"', '\\\"')

    # Extract entities from the response
    extraction_output = subprocess.run(["curl", "--header", "Content-Type: application/json", "--request", "POST",
                                        "--data", f"{{\"text\":\"{question_escaped} {response_escaped}\"}}",
                                        'https://labs.tib.eu/falcon/falcon2/api?mode=long'], capture_output=True, text=True)

    # Dictionary of entities and relations from extraction process
    try:
        dic_ents_rels = json.loads(extraction_output.stdout)
    except json.decoder.JSONDecodeError:
        dic_ents_rels = dict()

    print("Entities and Relations:", dic_ents_rels)
    print()

    # Feed question, LLM response, and entities and relations into LLM
    # Extract knowledge graph facts from the response
    llm_facts_json = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You will be provided with text and a dictionary of entities and relations, \
                and your task is to extract triples from the text in the form (subject URI, predicate URI, object URI)."
            },
            {
                "role": "user",
                "content": f"Text: {question} {response.strip()}\nEntities and Relations: {dic_ents_rels}"
            }
        ],
        temperature=0,
        max_tokens=256
    )

    print("LLM Facts JSON:", llm_facts_json)

    # Extract triples from the LLM extraction output
    triples = re.findall(r"\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*\)",
                         llm_facts_json["choices"][0]["message"]["content"], re.IGNORECASE)

    true_count = 0

    # For each triple, perform a SPARQL query to verify the truthfulness
    for triple in triples:
        print("Subject:", triple[0])
        print("Predicate:", triple[1])
        print("Object:", triple[2])

        # Convert the triple to SPARQL format
        if triple[0].startswith("http://"):
            subject = f"<{triple[0]}>"
        elif triple[0][1:-1].startswith("http://"):
            subject = f"<{triple[0][1:-1]}>"
        else:
            subject = triple[0]

        if triple[1].startswith("http://"):
            predicate = f"<{triple[1]}>"
        elif triple[1][1:-1].startswith("http://"):
            predicate = f"<{triple[1][1:-1]}>"
        else:
            predicate = triple[1]

        if triple[2].startswith("http://"):
            obj = f"<{triple[2]}>"
        elif triple[2][1:-1].startswith("http://"):
            obj = f"<{triple[2][1:-1]}>"
        else:
            obj = triple[2]

        sparql_query = f"ASK {{{subject} {predicate} {obj}.}}"

        print(sparql_query)

        # Perform the SPARQL query
        sparql_wd.setQuery(sparql_query)
        sparql_wd.setReturnFormat(JSON)
        sparql_result = sparql_wd.query().convert()
        print("Result:", sparql_result["boolean"], "\n")
        print()

        if sparql_result["boolean"]:
            true_count += 1

    print("True Count:", true_count)
    print()

    # Evaluate the truthfulness of the response

    '''
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
    '''

# Save the QA pairs in a JSON file
# with open("../output/qa_pairs.json", "w") as f:
#     json.dump(qa_pairs, f, indent=4)
