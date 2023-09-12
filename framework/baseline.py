import json
import subprocess

import openai
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from SPARQLWrapper import SPARQLWrapper, JSON
import spacy

import utilities.sparql_functions as sparql_f
import utilities.eval_metrics as eval_metrics
import utilities.llm_tasks_prompts as llm_tasks
import utilities.entity_link as el
import utilities.data_io as dio
from utilities.timeout import time_limit, TimeoutException

sparql_wd = SPARQLWrapper("https://query.wikidata.org/sparql")
# sparql_dbp = SPARQLWrapper("http://dbpedia.org/sparql")

# Add arguments to the command line
# < insert data options here >

# Load the data
data = dio.read_data("../data/mmlu_test/high_school_geography_test.csv")

# Create the language model
llm = OpenAI(temperature=0)

# Create a list of QA pairs
qa_pairs = dict()

num_correct = 0

# Generate a response for each question
for i, item in enumerate(data[66:68]):  # 66:71
    question = item['question_text']
    response = llm.predict(question)

    print("Q:", question)
    print("A:", response.strip(), "\n")

    qa_pairs[i] = dict()
    qa_pairs[i]["question"] = question
    qa_pairs[i]["choices"] = item['choices']
    qa_pairs[i]["initial_response"] = response.strip()

    new_response = ""
    try:
        with time_limit(120):
            question_escaped = question.replace('"', '\\\"')
            response_escaped = response.strip().replace('"', '\\\"')

            # Use the LLM to extract entities from the response
            entity_names = llm_tasks.extract_entities(f"{question} {response.strip()}")

            print(entity_names)
            num_of_identified_ents = len(entity_names)
            print("Number of entities identified:", num_of_identified_ents)
            print()

            qa_pairs[i]["entity_names"] = entity_names

            # Feed question, LLM response, and entities and relations into LLM
            # Extract knowledge graph facts from the response
            triples_names = llm_tasks.extract_kg_facts_given_entities(f"{question} {response.strip()}", entity_names)
            print("Triples:", triples_names)
            print()

            qa_pairs[i]["triples"] = triples_names

            entities_name_uri_map = dict()
            relations_name_uri_map = dict()

            uri_name_map = dict()

            extr_triples_uris = []

            # Convert the triples to a list of URIs
            for s, p, o in triples_names:
                # Use REST API to get the URI of the entity/property
                if s not in entities_name_uri_map:
                    entities_name_uri_map[s] = el.fetch_wikidata_from_query(s)
                if p not in relations_name_uri_map:
                    relations_name_uri_map[p] = el.fetch_wikidata_from_query(p, ent_type='property')
                if o not in entities_name_uri_map:
                    entities_name_uri_map[o] = el.fetch_wikidata_from_query(o)

                uris = []  # Used to construct the triple as a tuple of URIs

                for j, component in enumerate([s, p, o]):
                    # Retrieve the URI of the entity/property
                    if j == 1:
                        info = relations_name_uri_map[component]
                    else:
                        info = entities_name_uri_map[component]

                    if len(info['search']) == 0:
                        print('Sorry, no results for "' + component + '"')
                        uris.append('"' + component + '"')  # Use name of entity/property (quoted) instead
                        uri_name_map['"' + component + '"'] = component
                    else:
                        label = info['search'][0]["label"]
                        uri = info['search'][0]["concepturi"]
                        description = info['search'][0]["description"]
                        print(label, uri, description)
                        uris.append(uri)
                        uri_name_map[uri] = component
                print()

                extr_triples_uris.append(tuple(uris))

            # Print triples as tuple of URIs
            print("Extracted triples URIs:", extr_triples_uris)

            true_count = 0
            true_facts_uris = []
            true_facts_names = []
            true_entities_uris = set()

            # For each triple, perform a SPARQL query to verify the truthfulness
            for s, p, o in extr_triples_uris:
                # Convert the triple to SPARQL format
                s_format = sparql_f.uri_to_sparql_format_wikidata(s)
                p_format = sparql_f.uri_to_sparql_format_wikidata(p)
                o_format = sparql_f.uri_to_sparql_format_wikidata(o)

                # sparql_query = f"ASK {{{subject} {predicate} {obj}.}}"
                sparql_query = f"ASK {{{s_format} ?predicate {o_format}.}}"

                print(sparql_query)

                # Perform the SPARQL query
                sparql_result = sparql_f.execute_sparql_query(sparql_query, sparql_wd)
                print("Result:", sparql_result["boolean"], "\n")
                print()

                if sparql_result["boolean"]:
                    true_count += 1
                    true_facts_uris.append((s, p, o))
                    true_facts_names.append((uri_name_map[s], uri_name_map[p], uri_name_map[o]))
                    true_entities_uris.add(s)
                    true_entities_uris.add(o)

                else:
                    # Swap subject and object in case if the direction is incorrect
                    # sparql_query = f"ASK {{{obj} {predicate} {subject}.}}"
                    sparql_query = f"ASK {{{o_format} ?predicate {s_format}.}}"
                    print(sparql_query)

                    # Perform the SPARQL query
                    sparql_result = sparql_f.execute_sparql_query(sparql_query, sparql_wd)
                    print("Result:", sparql_result["boolean"], "\n")
                    print()

                    if sparql_result["boolean"]:
                        true_count += 1
                        true_facts_uris.append((o, p, s))
                        true_facts_names.append((uri_name_map[o], uri_name_map[p], uri_name_map[s]))
                        true_entities_uris.add(s)
                        true_entities_uris.add(o)

            print("True Count:", true_count)
            print("% True:", true_count / len(extr_triples_uris))
            print()

            print("True facts names:", true_facts_names)

            qa_pairs[i]["true_count"] = true_count
            qa_pairs[i]["% true"] = true_count / len(extr_triples_uris)

            # Calculate the number of linked entities
            linked_entities = set()
            for s, p, o in extr_triples_uris:
                linked_entities.add(s)
                linked_entities.add(o)
            print("Linked Entities:", linked_entities)
            print("Number of Linked Entities:", len(linked_entities))

            # Evaluate the truthfulness of the response
            eval_score = eval_metrics.simple_evaluation(entity_names, linked_entities,
                                                        extr_triples_uris, true_facts_uris)
            print("Evaluation Score:", eval_score)
            print()

            qa_pairs[i]["evaluation_score"] = eval_score

            if eval_score < 0.5:
                # Retrieve all entities from the true facts
                # true_entities = dict()
                # for j, (s_uri, _, o_uri) in enumerate(true_facts_uris):
                #     s_name, _, o_name = true_facts_names[j]
                #     true_entities[s_uri] = s_name
                #     true_entities[o_uri] = o_name
                #
                # true_relations = dict()
                # for j, (_, p_uri, _) in enumerate(true_facts_uris):
                #     true_relations[p_uri] = true_facts_names[j][1]

                print("True entities:", true_entities_uris)
                # print("True relations:", true_relations)
                print()

                quit()  # TODO: Remove quit placeholder

                # Do knowledge graph enrichment
                filtered_facts = []
                if len(true_entities_uris) > 0:  # TODO: Combine true entities with entities extracted from question
                    # Execute SPARQL query to get the list of predicate/object pairs
                    # subject = list(true_entities.keys())[0]
                    for subject in list(true_entities_uris):  # TODO: Check this line
                        print("Subject:", subject)
                        sparql_query = f'SELECT ?predicate WHERE {{{subject} ?predicate ?object. \
                        FILTER(!isLiteral(?object) || lang(?object) = "" || langMatches(lang(?object), "EN"))}}'
                        print(sparql_query)
                        sparql_output = sparql_f.execute_sparql_query(sparql_query, sparql_wd)
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
                            sparql_output = sparql_f.execute_sparql_query(sparql_query, sparql_wd)
                            sparql_bindings = sparql_output['results']['bindings']

                            # print(sparql_bindings)

                            for binding in sparql_bindings:
                                obj = binding['object']['value']
                                filtered_facts.append((subject, pred_uri, obj))

                        print()

                context_string = ""
                for s, p, o in filtered_facts:
                    s_name = uri_name_map[s]  # TODO: Check this line
                    p_name = sparql_f.get_name_from_dbpedia_uri(p)
                    o_name = sparql_f.get_name_from_dbpedia_uri(o)
                    context_string += f"{s_name} {p_name} {o_name}. "

                print("Context String:", context_string)

                qa_pairs[i]["context_string"] = context_string

                new_prompt = PromptTemplate(
                    input_variables=["question", "context"],
                    template="Question: {question}\nContext: {context}",
                )

                chain = LLMChain(llm=llm, prompt=new_prompt)
                new_response = chain.run({"question": question, "context": context_string})

                print("New Response:", new_response.strip())
            else:
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

    # new_response = llm.predict(f"{question}\nContext:{context_string}")

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

print("EM:", num_correct / len(data))

# Save the QA pairs in a JSON file
with open("../output/qa_sets_llm_kg_geography01.json", "w") as f:
    json.dump(qa_pairs, f, indent=4)

# 2 steps
# ASK {<http://www.wikidata.org/entity/Q652> ?predicate1 ?object1.
#      ?subject2 ?predicate2 <http://www.wikidata.org/entity/Q19814>.}
