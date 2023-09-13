import requests


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


def fetch_uri_wikidata(query, ent_type='item', limit=1):
    data = fetch_wikidata_from_query(query, ent_type=ent_type, limit=limit)
    if len(data['search']) == 0:
        print('Sorry, no results for "' + query + '"')
        return '"' + query + '"'
    else:
        label = data['search'][0]["label"]
        uri = data['search'][0]["concepturi"]
        description = data['search'][0]["description"]
        print(label, uri, description)
        return uri
