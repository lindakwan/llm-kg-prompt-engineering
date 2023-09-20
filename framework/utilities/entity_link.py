import requests
import utilities.llm_tasks_prompts as llm_tasks


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


def fetch_uri_wikidata(item_name, context, ent_type='item', limit=1):
    data = fetch_wikidata_from_query(item_name, ent_type=ent_type, limit=limit)
    if len(data['search']) == 0:
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
