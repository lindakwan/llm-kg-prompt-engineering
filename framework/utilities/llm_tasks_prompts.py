import openai
import re
import ast
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)


def extract_entities(text):
    """
    Extract the entity names from the text.
    :param text: The text to extract entities from.
    :return: The list of entity names extracted from the text.
    """
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
    extracted_entities = entities_json["choices"][0]["message"]["content"]
    entity_names = ast.literal_eval(extracted_entities)
    return entity_names


def extract_relations(text):
    relations_json = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You will be provided with text. \
                    Your task is to identify a list of predicate names mentioned in the text. No documentation, \
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
    extracted_relations = relations_json["choices"][0]["message"]["content"]
    relation_names = ast.literal_eval(extracted_relations)
    return relation_names


def extract_kg_facts(text, entities, relations):
    """
    Extract the triples from the text given the lists of entities and relations.
    :param text: The text to extract triples from.
    :param entities: The list of entities to choose from.
    :param relations: The list of relations to choose from.
    :return: The list of triples extracted from the text.
    """
    llm_facts_json = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You will be provided with text, list of entities, and list of relations. \
                    Using the lists of entities and relations, your task is to extract triples from the text in \
                    the form (subject, predicate, object)"  # (subject URI, predicate URI, object URI)."
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


def extract_kg_facts_given_entities(text, entities):
    """
    Extract the triples from the text given the lists of entities.
    :param text: The text to extract triples from.
    :param entities: The list of entities to choose from.
    :return: The list of triples extracted from the text.
    """
    llm_facts_json = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You will be provided with text and a list of entities. \
                    Using the lists of entities and relations, your task is to extract triples from the text in \
                    the form (subject, predicate, object)"
            },
            {
                "role": "user",
                "content": f"Text: {text}\nEntities: {entities}"
            }
        ],
        temperature=0,
        max_tokens=256
    )

    triples = re.findall(r"\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*\)",
                         llm_facts_json["choices"][0]["message"]["content"], re.IGNORECASE)

    return triples


def extract_relevant_predicates(text, predicates, k=3):
    """
    Extract the top k most relevant predicates to the text.
    :param text: The text to extract relevant predicates from.
    :param predicates: The list of predicates to choose from.
    :param k: The number of predicates to return.
    :return: The top k most relevant predicates to the text.
    """
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
        max_tokens=2000
    )

    relevant_preds_opt = relevant_preds_json["choices"][0]["message"]["content"]
    print("Relevant predicates output:", relevant_preds_opt)

    relevant_preds = ast.literal_eval(re.findall(r'\[.*?\]', relevant_preds_opt)[0])

    # Get the top k most relevant predicates
    top_preds = relevant_preds[:k]

    return top_preds


def select_mc_response_based(question, response, choices):
    """
    Select the best multiple choice response based on the question and response.
    :param question: The question fed into the model.
    :param response: The response generated by the model.
    :param choices: The multiple choice options.
    :return: The letter output of the best choice.
    """
    mc_prompt = PromptTemplate(
        input_variables=["question", "response", "choices"],
        template="Output the best one of the numbered options for the following question and response:\n \
                            Question: {question}\nResponse: {response}\nOptions:\n{choices}"
    )

    choices_text = "\n".join([str(i + 1) + ". " + choice for i, choice in enumerate(choices)])
    choice_response = llm(mc_prompt.format(question=question, response=response, choices=choices_text))

    print("Choice Response:", choice_response.strip())

    # Convert the response to the numbered choice
    numbers = [int(num) for num in re.findall(r'\d+', choice_response.strip().split(".")[0])]
    if len(numbers) == 0:
        # Choose A by default if no output
        numbered_output = 1
    else:
        numbered_output = numbers[-1]
    letter_output = chr(ord('A') + int(numbered_output) - 1)

    print("Generated answer:", letter_output)

    return letter_output
