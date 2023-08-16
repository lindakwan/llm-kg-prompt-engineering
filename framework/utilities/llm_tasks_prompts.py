import openai
import re
import ast


def extract_entities(text):
    entities_json = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You will be provided with text. \
                    Your task is to identify a list of entities mentioned in the text. No documentation, \
                    no explanation, only python3 list."
            },
            {
                "role": "user",
                "content": f"Text: {text}"
            }
        ],
        temperature=0,
        max_tokens=256
    )
    return entities_json["choices"][0]["message"]["content"]


def extract_kg_facts(text, entities, relations):
    llm_facts_json = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You will be provided with text, list of entities, and list of relations. \
                    Using the lists of entities and relations, your task is to extract triples from the text in \
                    the form (subject URI, predicate URI, object URI)."
            },
            {
                "role": "user",
                "content": f"Text: {text}\nEntities: {entities}\nRelations: {relations}"
            }
        ],
        temperature=0,
        max_tokens=256
    )

    triples = re.findall(r"\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*\)",
                         llm_facts_json["choices"][0]["message"]["content"], re.IGNORECASE)

    return triples


def extract_relevant_predicates(text, predicates, k=3):
    relevant_preds_json = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You will be provided with question text and a list of predicates. \
                Your task is to order the predicates by most relevant to text. No documentation, no explanation, \
                only python3 code."
            },
            {
                "role": "user",
                "content": f"Text: {text}\nPredicates: {predicates}"
            }
        ],
        temperature=0,
        max_tokens=1000
    )

    relevant_preds_opt = relevant_preds_json["choices"][0]["message"]["content"]
    print("Relevant predicates output:", relevant_preds_opt)

    relevant_preds = ast.literal_eval(re.findall(r'\[.*?\]', relevant_preds_opt)[0])

    # Get the top k most relevant predicates
    top_preds = relevant_preds[:k]

    return top_preds
