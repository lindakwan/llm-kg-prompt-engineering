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
import spacy

import utilities.sparql_functions as sparql_f
import utilities.eval_metrics as eval_metrics
import utilities.llm_tasks_prompts as llm_tasks

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

    # dbp_spotlight_output = nlp(question + " " + response.strip())
    # ent_list = [(ent.text, ent.kb_id_, ent._.dbpedia_raw_result['@similarityScore']) for ent in
    #             dbp_spotlight_output.ents]
    # ent_ids = [ent.kb_id_ for ent in dbp_spotlight_output.ents]

    # print("Doc:", dbp_spotlight_output)
    # print("Entities:", ent_list)

    question_escaped = question.replace('"', '\\\"')
    response_escaped = response.strip().replace('"', '\\\"')

    entity_names = llm_tasks.extract_entities(f"{question} {response.strip()}")

    print(entity_names)
    num_of_identified_ents = len(entity_names)
    print("Number of entities identified:", num_of_identified_ents)
    print()

    # Extract entities and relations from the response
    # Get top 5 results
    falcon_output = subprocess.run(["curl", "--header", "Content-Type: application/json", "--request", "POST",
                                    "--data", f"{{\"text\":\"{question_escaped} {response_escaped}\"}}",
                                    'https://labs.tib.eu/falcon/falcon2/api?mode=long&db=1&k=5'], capture_output=True, text=True)

    # Obtain list of relations from extraction process
    try:
        dic_ents_rels = json.loads(falcon_output.stdout)
        relations_dbpedia = dic_ents_rels['relations_dbpedia']
        entities_dbpedia = dic_ents_rels['entities_dbpedia']
    except json.decoder.JSONDecodeError:
        dic_ents_rels = dict()
        relations_dbpedia = []
        entities_dbpedia = []

    entities_ids = [ent['URI'] for ent in entities_dbpedia]
    print("Entities:", entities_ids)
    print()

    relations_ids = [rel['URI'] for rel in relations_dbpedia]
    print("Relations:", relations_ids)
    print()

    # Feed question, LLM response, and entities and relations into LLM
    # Extract knowledge graph facts from the response
    triples = llm_tasks.extract_kg_facts(f"{question} {response.strip()}", entities_ids, relations_ids)
    print("Triples:", triples)
    print()

    true_count = 0
    true_facts_uris = []
    true_facts_names = []

    # For each triple, perform a SPARQL query to verify the truthfulness
    for s, p, o in triples:
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
            true_facts_uris.append((subject, predicate, obj))
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
                true_facts_uris.append((obj, predicate, subject))
                true_facts_names.append((o_name, p_name, s_name))

    print("True Count:", true_count)
    print("% True:", true_count / len(triples))
    print()

    # Calculate the number of linked entities
    linked_entities = set()
    for s, p, o in triples:
        linked_entities.add(s)
        linked_entities.add(o)
    print("Linked Entities:", linked_entities)
    print("Number of Linked Entities:", len(linked_entities))

    # Evaluate the truthfulness of the response
    eval_score = eval_metrics.simple_evaluation(entity_names, linked_entities, triples, true_facts_uris)
    print("Evaluation Score:", eval_score)
    print()

    facts_sequence = ""

    for s_name, p_name, o_name in true_facts_names:
        facts_sequence += f"{s_name} {p_name} {o_name}. "

    print("Facts Sequence", facts_sequence)

#     facts_seq_length = len(facts_sequence.strip().split(" "))
#     response_length = len(response.strip().split(" "))
#
#     # Evaluate the truthfulness of the response
#     print("Length Facts Sequence / Length Response:", facts_seq_length / response_length)
#     print()

    true_entities = dict()
    for j, (s_uri, _, o_uri) in enumerate(true_facts_uris):
        s_name, _, o_name = true_facts_names[j]
        true_entities[s_uri] = s_name
        true_entities[o_uri] = o_name

    true_relations = dict()
    for j, (_, p_uri, _) in enumerate(true_facts_uris):
        true_relations[p_uri] = true_facts_names[j][1]

    print("True entities:", true_entities)
    print("True relations:", true_relations)
    print()

    # Do knowledge graph enrichment
    filtered_facts = []
    if len(true_entities) > 0:
        # Execute SPARQL query to get the list of predicate/object pairs
        # subject = list(true_entities.keys())[0]
        for subject in list(true_entities.keys()):
            print("Subject:", subject)
            sparql_query = f'SELECT ?predicate WHERE {{{subject} ?predicate ?object. \
            FILTER(!isLiteral(?object) || lang(?object) = "" || langMatches(lang(?object), "EN"))}}'
            print(sparql_query)
            sparql_output = sparql_f.execute_sparql_query(sparql_query, sparql_dbp)
            sparql_bindings = sparql_output['results']['bindings']

            unique_predicates = dict()

            for binding in sparql_bindings:
                predicate = binding['predicate']['value']
                predicate_alias = sparql_f.get_name_from_dbpedia_uri(predicate)
                if predicate_alias not in unique_predicates:
                    unique_predicates[predicate_alias] = []
                unique_predicates[predicate_alias].append(predicate)

            # print("Unique predicates:", unique_predicates.keys())

            # Given a list of predicates, use the LLM to get the order of predicates by most relevant
            top_preds = llm_tasks.extract_relevant_predicates(question, list(unique_predicates.keys()), k=3)

            print("Top predicates:", top_preds)
            print()

            # Execute SPARQL query for each of the top 5 predicates
            for pred in top_preds:
                pred_uri = unique_predicates[pred][0]
                sparql_query = f'SELECT ?object WHERE {{{subject} <{pred_uri}> ?object. \
                FILTER(!isLiteral(?object) || lang(?object) = "" || langMatches(lang(?object), "EN"))}}'
                print(sparql_query)
                sparql_output = sparql_f.execute_sparql_query(sparql_query, sparql_dbp)
                sparql_bindings = sparql_output['results']['bindings']

                # print(sparql_bindings)

                for binding in sparql_bindings:
                    obj = binding['object']['value']
                    filtered_facts.append((subject, pred_uri, obj))

    context_string = ""
    for s, p, o in filtered_facts:
        s_name = true_entities[s]
        p_name = sparql_f.get_name_from_dbpedia_uri(p)
        o_name = sparql_f.get_name_from_dbpedia_uri(o)
        context_string += f"{s_name} {p_name} {o_name}. "

    print("Context String:", context_string)

    new_response = llm.predict(f"{question}\nContext:{context_string}")

    print("New Response:", new_response.strip())

        # relevant_entities_json = openai.ChatCompletion.create(
        #     model="gpt-3.5-turbo",
        #     messages=[
        #         {
        #             "role": "system",
        #             "content": "You will be provided with question text and a list of entities. \
        #             Your task is to order the entities by most relevant to text."
        #         },
        #         {
        #             "role": "user",
        #             "content": f"Text: {question}\nEntities: {unique_entities.keys()}"
        #         }
        #     ],
        #     temperature=0,
        #     max_tokens=256
        # )
        #
        # print(relevant_entities_json)

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
