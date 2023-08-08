import os
import json
import csv
import re
import subprocess

import openai
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from SPARQLWrapper import SPARQLWrapper, JSON
import spacy

# OpenAI API key
openai_api_key_file = open("../openai_api_key.txt", "r")
os.environ["OPENAI_API_KEY"] = openai_api_key_file.read().strip()
openai_api_key_file.close()

# sparql_wd = SPARQLWrapper("https://query.wikidata.org/sparql")
sparql_dbp = SPARQLWrapper("http://dbpedia.org/sparql")


def get_name_from_dbpedia_uri(uri):
    return uri.split("/")[-1].replace("_", " ")


def execute_sparql_query(query, endpoint):
    """
    Perform a SPARQL query
    :param query: The SPARQL query
    :param endpoint: The SPARQL endpoint
    :return: Output of the query
    """
    endpoint.setQuery(query)
    endpoint.setReturnFormat(JSON)
    results = endpoint.query().convert()
    return results


def uri_to_sparql_format(uri):
    if uri.startswith("http://"):
        return f"<{uri}>", get_name_from_dbpedia_uri(uri)
    elif uri[1:-1].startswith("http://"):
        return f"<{uri[1:-1]}>", get_name_from_dbpedia_uri(uri[1:-1])
    else:
        return uri, uri


# Load the data
file = open("../data/mmlu_test/high_school_geography_test.csv", "r")
csv_reader = csv.reader(file, delimiter=',')
data = []
for row in csv_reader:
    data.append({"question_text": row[0], "choices": row[1:-1], "correct_answer": row[-1]})

# Create the language model
llm = OpenAI(temperature=0)

# Create NLP model for extracting entities
nlp = spacy.blank("en")
nlp.add_pipe("dbpedia_spotlight")

# Create a list of QA pairs
# qa_pairs = dict()

# Generate a response for each question
for i, item in enumerate(data[66:70]):  # 66:71
    question = item['question_text']
    response = llm.predict(question)

    print("Q:", question)
    print("A:", response.strip(), "\n")

    # qa_pairs[i] = dict()
    # qa_pairs[i]["question"] = question
    # qa_pairs[i]["response"] = response.strip()

    dbp_spotlight_output = nlp(question + " " + response.strip())
    ent_list = [(ent.text, ent.kb_id_, ent._.dbpedia_raw_result['@similarityScore']) for ent in
                dbp_spotlight_output.ents]
    ent_ids = [ent.kb_id_ for ent in dbp_spotlight_output.ents]

    print("Doc:", dbp_spotlight_output)
    print("Entities:", ent_list)

    question_escaped = question.replace('"', '\\\"')
    response_escaped = response.strip().replace('"', '\\\"')

    # Extract relations from the response
    falcon_output = subprocess.run(["curl", "--header", "Content-Type: application/json", "--request", "POST",
                                    "--data", f"{{\"text\":\"{question_escaped} {response_escaped}\"}}",
                                    'https://labs.tib.eu/falcon/falcon2/api?mode=long&db=1'], capture_output=True, text=True)
    # Maybe get top k results instead

    # Obtain list of relations from extraction process
    try:
        dic_ents_rels = json.loads(falcon_output.stdout)
        relations_dbpedia = dic_ents_rels['relations_dbpedia']
    except json.decoder.JSONDecodeError:
        dic_ents_rels = dict()
        relations_dbpedia = []

    relation_ids = [rel['URI'] for rel in relations_dbpedia]
    print("Relations:", relations_dbpedia)

    # Feed question, LLM response, and entities and relations into LLM
    # Extract knowledge graph facts from the response
    llm_facts_json = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You will be provided with text, list of entities, and list of relations. \
                Your task is to extract triples from the text in the form (subject URI, predicate URI, object URI)."
            },
            {
                "role": "user",
                "content": f"Text: {question} {response.strip()}\nEntities: {ent_ids}\nRelations: {relation_ids}"
            }
        ],
        temperature=0,
        max_tokens=256
    )

    print("LLM Facts JSON:", llm_facts_json)

    # Extract triples from the LLM extraction output
    triples = re.findall(r"\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*\)",
                         llm_facts_json["choices"][0]["message"]["content"], re.IGNORECASE)

    print(triples)
    print()

    true_count = 0
    true_facts = []
    true_facts_names = []

    # For each triple, perform a SPARQL query to verify the truthfulness
    for s, p, o in triples:
        print("Subject:", s)
        print("Predicate:", p)
        print("Object:", o)

        # Convert the triple to SPARQL format
        subject, s_name = uri_to_sparql_format(s)
        predicate, p_name = uri_to_sparql_format(p)
        obj, o_name = uri_to_sparql_format(o)

        sparql_query = f"ASK {{{subject} {predicate} {obj}.}}"

        print(sparql_query)

        # Perform the SPARQL query
        sparql_result = execute_sparql_query(sparql_query, sparql_dbp)
        print("Result:", sparql_result["boolean"], "\n")
        print()

        if sparql_result["boolean"]:
            true_count += 1
            true_facts.append((subject, predicate, obj))
            true_facts_names.append((s_name, p_name, o_name))
        else:
            # Swap subject and object in case if the direction is incorrect
            sparql_query = f"ASK {{{obj} {predicate} {subject}.}}"
            print(sparql_query)

            # Perform the SPARQL query
            sparql_result = execute_sparql_query(sparql_query, sparql_dbp)
            print("Result:", sparql_result["boolean"], "\n")
            print()

            if sparql_result["boolean"]:
                true_count += 1
                true_facts.append((obj, predicate, subject))
                true_facts_names.append((o_name, p_name, s_name))

    print("True Count:", true_count)
    print("% True:", true_count / len(triples))
    print()

    facts_sequence = ""

    for s, p, o in true_facts_names:
        facts_sequence += f"{s} {p} {o}."

    print("Facts Sequence", facts_sequence)

    facts_seq_length = len(facts_sequence.strip().split(" "))
    response_length = len(response.strip().split(" "))

    print("Length Facts Sequence / Length Response:", facts_seq_length / response_length)
    print()

    true_entities = []
    for s, p, o in true_facts:
        true_entities.append(s)
        true_entities.append(o)

    print(true_entities)
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
