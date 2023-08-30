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


def fetch_wikidata_from_query(query, ent_type='item', limit=5):
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
