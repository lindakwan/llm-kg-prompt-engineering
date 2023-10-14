import requests
import utilities.llm_tasks_prompts as llm_tasks
import utilities.emb_tasks as emb_tasks


def fetch_wikidata(params):
    url = 'https://www.wikidata.org/w/api.php'
    try:
        return requests.get(url, params=params)
    except:
        return 'There was and error'


def generate_parameters(query, ent_type='item', limit=5):
    """
    Generate parameters for the wikidata API
    :param query: entity or property to match
    :param ent_type: 'item' (by default) or 'property'
    :param limit: maximum number of results to return
    :return: parameters for the wikidata API
    """
    params = {
        'action': 'wbsearchentities',
        'format': 'json',
        'search': query,
        'language': 'en',
        'type': ent_type,
        'limit': limit
    }
    return params


def fetch_wikidata_from_query(query, ent_type='item', limit=1):
    """
    Use wikidata API to find matching URIs for an entity or property
    :param query: entity or property to match
    :param ent_type: 'item' (by default) or 'property'
    :param limit: maximum number of results to return
    :return: the query results
    """
    params = generate_parameters(query, ent_type=ent_type, limit=limit)
    data = fetch_wikidata(params)
    data = data.json()
    return data


def condense_wikidata_results(data):
    """
    Condense the wikidata API results into a dictionary
    :param data: the query results
    :return: a dictionary of the query results
    """
    results = []
    for match in data['search']:
        results.append({
            'label': match['label'],
            'uri': match['concepturi'],
            'description': match.get("description", "No description available.")
        })
    return results


def fetch_uri_wikidata_simple(item_name, context, ent_type='item', limit=1):
    data = fetch_wikidata_from_query(item_name, ent_type=ent_type, limit=limit)
    if ('error' in data) or (len(data['search']) == 0):
        print('Sorry, no results for "' + item_name + '" from REST API')
        if ent_type == 'item':
            uri = llm_tasks.get_similar_identifier_given_context(item_name, context, item_type='item')
            print(item_name, uri)
            return uri
        elif ent_type == 'property':
            uri = llm_tasks.get_similar_identifier_given_context(item_name, context, item_type='property')
            print(item_name, uri)
            return uri
        else:
            return '"' + item_name + '"'  # TODO: check this
    else:
        label = data['search'][0]["label"]

        if ent_type == 'item':
            uri = "wd:" + data['search'][0]["id"]
        elif ent_type == 'property':
            uri = "wdt:" + data['search'][0]["id"]
        else:
            uri = data['search'][0]["concepturi"]

        description = data['search'][0].get("description", "No description available.")
        print(label, uri, description)
        return uri


def fetch_uri_wikidata(item_name, context, ent_type='item', limit=3):
    """
    Fetch the URI for an entity or property from the wikidata API
    :param item_name: Name of the entity or property
    :param context: The context of the item name in the form "<subject> <predicate> <object>."
    :param ent_type: Either 'item' or 'property'
    :param limit: The maximum number of results to return from the API
    :return: The URI for the entity or property, along with its label and description, as a tuple
    """

    data = fetch_wikidata_from_query(item_name, ent_type=ent_type, limit=limit)

    if ('error' in data) or (len(data['search']) == 0):
        print('Sorry, no results for "' + item_name + '" from REST API')
        if ent_type == 'item':
            # Use the LLM instead to find a similar entity
            uri = llm_tasks.get_similar_identifier_given_context(item_name, context, item_type='item')
            print(item_name, uri, None)
            return uri, None, item_name
        elif ent_type == 'property':
            # Use the LLM instead to find a similar property
            uri = llm_tasks.get_similar_identifier_given_context(item_name, context, item_type='property')
            print(item_name, uri, None)
            return uri, None, item_name
        else:
            return '"' + item_name + '"', None, item_name  # TODO: check this
    else:
        matches = []
        ext_descriptions = []

        for i in range(len(data['search'])):
            # Get the label for the entity or property
            label = data['search'][i]["label"]

            # Get the URI for the entity or property
            if ent_type == 'item':
                uri = "wd:" + data['search'][i]["id"]
            elif ent_type == 'property':
                uri = "wdt:" + data['search'][i]["id"]
            else:
                uri = data['search'][i]["concepturi"]

            # Get the description for the entity or property
            description = data['search'][i].get("description", None)
            print(label, uri, description)

            matches.append((uri, description, label))
            if description is not None:
                ext_descriptions.append(label + " is " + description)
            else:
                ext_descriptions.append(label)

        # Calculate the cosine similarities between the context and all the descriptions
        cos_sims = emb_tasks.calculate_cos_sim_multiple(context, ext_descriptions)
        print("Cosine similarities:", cos_sims)

        best_match = matches[cos_sims.argmax()]
        print("Best match:", best_match, '\n')

        return best_match


def fetch_uri_or_none_wikidata(item_name, context, ent_type='item', limit=3):
    """
    Fetch the URI for an entity or property from the wikidata API
    :param item_name: Name of the entity or property
    :param context: The context of the item name in the form "<subject> <predicate> <object>."
    :param ent_type: Either 'item' or 'property'
    :param limit: The maximum number of results to return from the API
    :return: The URI for the entity or property, along with its label and description, as a tuple
    """

    data = fetch_wikidata_from_query(item_name, ent_type=ent_type, limit=limit)

    if ('error' in data) or (len(data['search']) == 0):
        print('Sorry, no results for "' + item_name + '" from REST API')
        return None
    else:
        matches = []
        ext_descriptions = []

        for i in range(len(data['search'])):
            # Get the URI for the entity or property
            uri = data['search'][i]["concepturi"]

            # Get the label for the entity or property
            label = data['search'][i]["label"]

            # Get the description for the entity or property
            description = data['search'][i].get("description", None)
            print(label, uri, description)

            matches.append((uri, label, description))
            if description is not None:
                ext_descriptions.append(label + " is " + description)
            else:
                ext_descriptions.append(label)

        # Calculate the cosine similarities between the context and all the descriptions
        cos_sims = emb_tasks.calculate_cos_sim_multiple(context, ext_descriptions)
        print("Cosine similarities:", cos_sims)

        best_match = matches[cos_sims.argmax()]
        print("Best match:", best_match)

        return best_match
