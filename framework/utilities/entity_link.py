import requests


def fetch_wikidata(params):
    url = 'https://www.wikidata.org/w/api.php'
    try:
        return requests.get(url, params=params)
    except:
        return 'There was and error'


def generate_parameters(query):
    params = {
        'action': 'wbsearchentities',
        'format': 'json',
        'search': query,
        'language': 'en'
    }
    return params


def fetch_wikidata_from_query(query):
    params = generate_parameters(query)
    data = fetch_wikidata(params)
    data = data.json()
    return data
