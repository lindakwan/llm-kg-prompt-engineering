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

import framework.utilities.sparql_functions as sparql_f

# OpenAI API key
openai_api_key_file = open("../openai_api_key.txt", "r")
os.environ["OPENAI_API_KEY"] = openai_api_key_file.read().strip()
openai_api_key_file.close()

# sparql_wd = SPARQLWrapper("https://query.wikidata.org/sparql")
sparql_dbp = SPARQLWrapper("http://dbpedia.org/sparql")

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
for i, item in enumerate(data[67:68]):  # 66:71
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

    # print("LLM Facts JSON:", llm_facts_json)

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
        # print("Subject:", s)
        # print("Predicate:", p)
        # print("Object:", o)

        # Convert the triple to SPARQL format
        subject, s_name = sparql_f.uri_to_sparql_format(s)
        predicate, p_name = sparql_f.uri_to_sparql_format(p)
        obj, o_name = sparql_f.uri_to_sparql_format(o)

        sparql_query = f"ASK {{{subject} {predicate} {obj}.}}"

        print(sparql_query)

        # Perform the SPARQL query
        sparql_result = sparql_f.execute_sparql_query(sparql_query, sparql_dbp)
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
            sparql_result = sparql_f.execute_sparql_query(sparql_query, sparql_dbp)
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
        facts_sequence += f"{s} {p} {o}. "

    print("Facts Sequence", facts_sequence)

    facts_seq_length = len(facts_sequence.strip().split(" "))
    response_length = len(response.strip().split(" "))

    # Evaluate the truthfulness of the response
    print("Length Facts Sequence / Length Response:", facts_seq_length / response_length)
    print()

    true_entities = set()
    for s, p, o in true_facts:
        true_entities.add(s)
        true_entities.add(o)

    true_relations = set()
    for s, p, o in true_facts:
        true_relations.add(p)

    print("True entities:", true_entities)
    print("True relations:", true_relations)
    print()

    # Do knowledge graph enrichment
    if len(true_entities) > 0:
        sparql_query = f'SELECT ?predicate ?object WHERE {{{list(true_entities)[-1]} ?predicate ?object. \
        FILTER(!isLiteral(?object) || lang(?object) = "" || langMatches(lang(?object), "EN"))}}'
        print(sparql_query)
        sparql_result = sparql_f.execute_sparql_query(sparql_query, sparql_dbp)
        print(sparql_result)
        print()

    '''
    # Example SPARQL Query
    SELECT ?subject ?predicate ?object
    WHERE
    {
        wd: Q1299 ?predicate ?object.
    }
    
    SELECT ?predicate ?object
    WHERE {
        <http://dbpedia.org/resource/Romance_languages> ?predicate ?object.
        FILTER((!isLiteral(?object) || lang(?object) = "" || langMatches(lang(?object), "EN")) 
        && (regex(str(?predicate), "http://dbpedia.org/property/")
        || regex(str(?predicate), "http://dbpedia.org/ontology/")))
}
    '''

# Save the QA pairs in a JSON file
# with open("../output/qa_pairs.json", "w") as f:
#     json.dump(qa_pairs, f, indent=4)
