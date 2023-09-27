import json
import argparse
import datetime

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from SPARQLWrapper import SPARQLWrapper

import utilities.sparql_functions as sparql_f
import utilities.eval_metrics as eval_metrics
import utilities.llm_tasks_prompts as llm_tasks
import utilities.entity_link as el
import utilities.data_io as dio
import utilities.nlp_tasks as nlp_tasks
import utilities.emb_tasks as emb_tasks
from utilities.timeout import time_limit, TimeoutException

# Create a list of QA pairs
qa_pairs = dict()

start_time = datetime.datetime.now()

qa_pairs['start_time'] = str(start_time)

sparql_wd = SPARQLWrapper("https://query.wikidata.org/sparql")
# sparql_dbp = SPARQLWrapper("http://dbpedia.org/sparql")

# Add arguments to the command line
parser = argparse.ArgumentParser(description='Run the combined LLM-KG on a dataset.')
parser.add_argument('-d', '--dataset', help='The dataset to run the LLM-KG on.', required=True)
args = parser.parse_args()

dataset_path = ""
json_output_path = ""

if args.dataset == "geography":
    dataset_path = "../data/mmlu_test/high_school_geography_test_filtered.csv"
    # json_output_path = f"../output/qa_sets_llm_kg_geography_wd_{start_time.timestamp()}.json"
    json_output_path = f"../output/qa_sets_llm_kg_geography01.json"
elif args.dataset == "government_and_politics":
    dataset_path = "../data/mmlu_test/high_school_government_and_politics_test_filtered.csv"
    # json_output_path = f"../output/qa_sets_llm_kg_government_and_politics_wd_{start_time.timestamp()}.json"
    json_output_path = f"../output/qa_sets_llm_kg_government_and_politics01.json"
elif args.dataset == "miscellaneous":
    dataset_path = "../data/mmlu_test/miscellaneous_test_filtered.csv"
    json_output_path = f"../output/qa_sets_llm_kg_miscellaneous_wd_{start_time.timestamp()}.json"
else:
    print("Invalid dataset.")
    exit()

# Load the data
data = dio.read_data(dataset_path)

# Create the language model
llm = OpenAI(temperature=0)

num_correct = 0

# Generate a response for each question
for i, item in enumerate(data[41:42]):  # 41:42
    question = item['question_text']
    # response = llm_tasks.generate_response_with_elaboration(question)
    response = llm.predict(question)

    print("Q:", question)
    print("A:", response.strip(), "\n")

    qa_pairs[i] = dict()
    qa_pairs[i]["question"] = question
    qa_pairs[i]["choices"] = item['choices']
    qa_pairs[i]["initial_response"] = response.strip()

    new_response = ""
    try:
        with time_limit(180):
            # Use the LLM to extract entities from the question
            question_entities = llm_tasks.extract_entities(question)
            print("Q entities:", question_entities)

            # Use the LLM to extract entities from the response
            response_entities = llm_tasks.extract_entities(response.strip())
            print("A entities:", response_entities)

            # Combine the entities from the question and response
            entity_names = list(dict.fromkeys(question_entities + response_entities))
            print("All entities:", entity_names)

            qa_pairs[i]["question_entity_names"] = question_entities
            qa_pairs[i]["response_entity_names"] = response_entities

            # Feed question, LLM response, and entities into LLM and extract knowledge graph facts from the response
            triples_names = llm_tasks.extract_kg_facts_given_entities(f"{question} {response.strip()}", entity_names)
            triples_names = nlp_tasks.remove_stopwords_from_triples(triples_names)
            print("Triples:", triples_names)
            print()

            qa_pairs[i]["extracted_triples"] = triples_names

            name_uri_map = dict()
            uri_name_map = dict()

            extr_triples_uris = []

            # Convert the triples to a list of URIs
            for triple in triples_names:
                # TODO: Include description as well
                context_str = f"{triple[0]} {triple[1]} {triple[2]}"

                uris = []  # Used to construct the triple as a tuple of URIs

                # Get the URI for each component of the triple
                for j, component in enumerate(triple):
                    # Use REST API to retrieve the URI of the entity/property
                    if j == 1:
                        if (component, 'rel') not in name_uri_map:
                            uri = el.fetch_uri_wikidata_simple(component, context_str, ent_type='property')
                            name_uri_map[(component, 'rel')] = uri
                            uri_name_map[uri] = component
                        else:
                            uri = name_uri_map[(component, 'rel')]
                    else:
                        if (component, 'ent') not in name_uri_map:
                            uri = el.fetch_uri_wikidata_simple(component, context_str)
                            name_uri_map[(component, 'ent')] = uri
                            uri_name_map[uri] = component
                        else:
                            uri = name_uri_map[(component, 'ent')]
                    uris.append(uri)
                print()

                extr_triples_uris.append(tuple(uris))

            # Print triples as tuple of URIs
            print("Extracted triples URIs:", extr_triples_uris, '\n')
            qa_pairs[i]["extracted_triples_uris"] = extr_triples_uris

            # Get the URIs for the entities in the question
            question_ent_uris = []
            for ent_name in question_entities:
                if (ent_name, 'ent') not in name_uri_map:
                    ent_uri = el.fetch_uri_wikidata_simple(ent_name, ent_name)
                    name_uri_map[(ent_name, 'ent')] = ent_uri
                    uri_name_map[ent_uri] = ent_name
                question_ent_uris.append(name_uri_map[(ent_name, 'ent')])

            print("Question entities URIs:", question_ent_uris)
            print()

            qa_pairs[i]["uri_name_map"] = uri_name_map

            truth_score = 0
            true_facts_uris = []
            true_facts_names = []
            true_entities_uris = set()

            # For each triple, perform a SPARQL query to verify the truthfulness
            for s, p, o in extr_triples_uris:
                print("Triple:", s, p, o)

                p_uri_label_pairs, o_uri_label_pairs = sparql_f.get_sparql_results_wikidata(s, p, o)

                print("Predicate URI label pairs:", p_uri_label_pairs)
                print("Object URI label pairs:", o_uri_label_pairs)

                best_sim_score = 0
                best_fact_with_names = None
                best_fact_with_uris = None

                for p_uri, p_label in p_uri_label_pairs:
                    print("Predicate URI-label pair:", p_uri, p_label)
                    sim_score = emb_tasks.calculate_cos_sim(uri_name_map[p], p_label)
                    print("Predicate similarity score:", sim_score)
                    if sim_score > best_sim_score:
                        best_sim_score = sim_score
                        best_fact_with_names = (uri_name_map[s], p_label, uri_name_map[o])
                        best_fact_with_uris = (s, p_uri, o)

                for o_uri, o_label in o_uri_label_pairs:
                    print("Object URI-label pair:", o_uri, o_label)
                    sim_score = emb_tasks.calculate_cos_sim(uri_name_map[o], o_label)
                    print("Object similarity score:", sim_score)
                    if sim_score > best_sim_score:
                        best_sim_score = sim_score
                        best_fact_with_names = (uri_name_map[s], uri_name_map[p], o_label)
                        best_fact_with_uris = (s, p, o_uri)

                print("Best similarity score:", best_sim_score)
                print("Best fact with names:", best_fact_with_names)
                print("Best fact with URIs:", best_fact_with_uris)

                if best_sim_score > 0:
                    truth_score += best_sim_score
                    true_facts_uris.append(best_fact_with_uris)
                    true_facts_names.append(best_fact_with_names)
                    true_entities_uris.add(best_fact_with_uris[0])
                    true_entities_uris.add(best_fact_with_uris[2])

                    uri_name_map[best_fact_with_uris[0]] = best_fact_with_names[0]
                    uri_name_map[best_fact_with_uris[1]] = best_fact_with_names[1]
                    uri_name_map[best_fact_with_uris[2]] = best_fact_with_names[2]

            print("Truth Score:", truth_score)

            if len(extr_triples_uris) > 0:
                frac_true = truth_score / len(extr_triples_uris)
            else:
                frac_true = 0
            print("Simple measure of truthfulness:", frac_true)

            print("True facts names:", true_facts_names)

            qa_pairs[i]["truth_score"] = truth_score
            qa_pairs[i]["% true"] = frac_true

            qa_pairs[i]["true_facts_names"] = true_facts_names
            qa_pairs[i]["true_facts_uris"] = true_facts_uris

            # Calculate the number of linked entities
            linked_entities = set()
            for s, p, o in extr_triples_uris:
                if s.startswith("wd:"):
                    linked_entities.add(s)
                if o.startswith("wd:"):
                    linked_entities.add(o)
            print("Linked Entities:", linked_entities)
            print("Number of Linked Entities:", len(linked_entities))

            qa_pairs[i]["linked_entities"] = list(linked_entities)

            # Evaluate the truthfulness of the response
            eval_score = eval_metrics.simple_evaluation_using_similarity(entity_names, linked_entities,
                                                                         triples_names, truth_score)
            print("Evaluation Score:", eval_score)
            print()

            qa_pairs[i]["evaluation_score"] = eval_score

            if eval_score < 0.8:
                qa_pairs[i]["below_threshold"] = True

                print("True entities:", true_entities_uris)
                # print("True relations:", true_relations)
                print()

                qa_pairs[i]["true_entities"] = list(true_entities_uris)

                # Do knowledge graph enrichment
                filtered_facts = []

                # Combine true entities with entities extracted from question
                focus_entities = true_entities_uris.union(question_ent_uris)

                if len(focus_entities) > 0:
                    # Execute SPARQL query to get the list of predicate/object pairs for each subject
                    for subject in list(focus_entities):
                        s_format = sparql_f.uri_to_sparql_format_wikidata(subject)
                        print("Subject:", subject)

                        sparql_query = f'''
                        SELECT ?predicate ?propLabel ?object ?objectLabel WHERE {{
                            {s_format} ?predicate ?object.
                            SERVICE wikibase:label {{
                                bd:serviceParam wikibase:language "en" . }}
                            ?prop wikibase:directClaim ?predicate .
                            ?prop rdfs:label ?propLabel.  filter(lang(?propLabel) = "en").
                            FILTER(!isLiteral(?object) || lang(?object) = "" || langMatches(lang(?object), "EN")) }}'''

                        print(sparql_query)
                        sparql_output = sparql_f.execute_sparql_query(sparql_query, sparql_wd)
                        sparql_bindings = sparql_output['results']['bindings']

                        # print("SPARQL bindings:", sparql_bindings)

                        unique_predicates = dict()  # Map alias to URI

                        # Get uris and names of unique predicates
                        for binding in sparql_bindings:
                            predicate_uri = binding['predicate']['value']
                            predicate_name = binding['propLabel']['value']
                            if predicate_name not in unique_predicates:
                                unique_predicates[predicate_name] = []
                            if predicate_uri not in unique_predicates[predicate_name]:
                                unique_predicates[predicate_name].append(predicate_uri)

                        # print("Unique predicates:", unique_predicates)
                        # print()

                        # qa_pairs[i]["unique_predicates_" + uri_name_map[subject]] = unique_predicates

                        # Given a list of predicates, use the LLM to get the order of predicates by most relevant
                        top_preds = []
                        unique_pred_names = list(unique_predicates.keys())
                        start_idx = 0

                        while start_idx < len(unique_pred_names):
                            end_idx = min(start_idx + 80, len(unique_pred_names))
                            top_preds += llm_tasks.extract_relevant_predicates(question, unique_pred_names[start_idx:end_idx], k=3)
                            start_idx += 80

                        if len(top_preds) > 3:
                            top_preds = llm_tasks.extract_relevant_predicates(question, top_preds, k=3)

                        print("Top predicates:", top_preds)
                        print()

                        qa_pairs[i]["top_predicates_" + uri_name_map[subject]] = top_preds

                        # Execute SPARQL query for each of the top 3 predicates
                        for top_pred in top_preds:
                            # pred_uri = unique_predicates[top_pred][0]
                            pred_uri = el.fetch_uri_wikidata_simple(top_pred, top_pred, ent_type='property')
                            top_p_format = sparql_f.uri_to_sparql_format_wikidata(pred_uri)
                            # TODO: Use the existing SPARQL binding instead of executing a new SPARQL query
                            sparql_query = f'''
                            SELECT ?object ?objectLabel WHERE {{
                                {s_format} {top_p_format} ?object.
                                SERVICE wikibase:label {{
                                    bd:serviceParam wikibase:language "en" . }}
                                FILTER(!isLiteral(?object) || lang(?object) = "" || langMatches(lang(?object), "EN"))
                            }}'''
                            print(sparql_query)
                            sparql_output = sparql_f.execute_sparql_query(sparql_query, sparql_wd)
                            sparql_bindings = sparql_output['results']['bindings']

                            for binding in sparql_bindings:
                                objLabel = binding['objectLabel']['value']
                                filtered_facts.append((uri_name_map[subject], top_pred, objLabel))

                print(filtered_facts)
                print()

                # TODO: Split filtered facts into batches

                qa_pairs[i]["filtered_facts_for_context"] = filtered_facts

                context_string = ""
                for s_name, p_name, o_name in filtered_facts:
                    context_string += f"{s_name} {p_name} {o_name}. "

                print("Context String:", context_string)

                qa_pairs[i]["context_string"] = context_string

                new_prompt = PromptTemplate(
                    input_variables=["question", "context"],
                    template="Question: {question}\nContext: {context}",
                )

                chain = LLMChain(llm=llm, prompt=new_prompt)
                new_response = chain.run({"question": question, "context": context_string})

                # new_response = llm_tasks.generate_response_using_context_with_elaboration(question, context_string)

                print("New Response:", new_response.strip())

                qa_pairs[i]["new_response"] = new_response.strip()
            else:
                qa_pairs[i]["below_threshold"] = False
                new_response = response.strip()
    except TimeoutException:
        print("Timeout Exception")
        new_response = response.strip()
        qa_pairs[i]["timeout"] = True
    except Exception as exc:
        print("Error:", exc)
        new_response = response.strip()
        qa_pairs[i]["error"] = str(exc)

    # Generate the letter output based on the response
    letter_output = llm_tasks.select_mc_response_based(question, new_response.strip(), item['choices'])

    print("Generated answer:", letter_output)

    # Evaluate the response
    is_correct = letter_output == item['correct_answer']
    print("Correct answer:", item['correct_answer'])
    print("Correct:", is_correct, "\n")

    # Update the number of correct answers
    if letter_output == item['correct_answer']:
        num_correct += 1

    qa_pairs[i]["llm_answer"] = letter_output
    qa_pairs[i]["llm_is_correct"] = is_correct

    with open(json_output_path, "w") as f:
        json.dump(qa_pairs, f, indent=4)

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

em = num_correct / len(data)
print("EM:", em)

qa_pairs['finish_time'] = str(datetime.datetime.now())
qa_pairs['EM'] = em

# Save the QA pairs in a JSON file
with open(json_output_path, "w") as f:
    json.dump(qa_pairs, f, indent=4)

# 2 steps
# ASK {<http://www.wikidata.org/entity/Q652> ?predicate1 ?object1.
#      ?subject2 ?predicate2 <http://www.wikidata.org/entity/Q19814>.}

# Get predicate label
# SELECT ?predicate ?propLabel ?object ?objectLabel WHERE {
#   <http://www.wikidata.org/entity/Q43473> ?predicate ?object.
#   SERVICE wikibase:label {
#     bd:serviceParam wikibase:language "en" .
#   }
#   ?prop wikibase:directClaim ?predicate .
#   ?prop rdfs:label ?propLabel.  filter(lang(?propLabel) = "en").
#   FILTER(!isLiteral(?object) || lang(?object) = "" || langMatches(lang(?object), "EN"))
# }

# Get description
# SELECT ?p ?pLabel ?pDescription ?w ?wLabel ?wDescription WHERE {
#    wd:Q30 p:P6/ps:P6 ?p .
#    ?p wdt:P26 ?w .
#    SERVICE wikibase:label {
#     bd:serviceParam wikibase:language "en" .
#    }
# }
