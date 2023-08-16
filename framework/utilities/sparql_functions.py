from SPARQLWrapper import JSON


def get_name_from_dbpedia_uri(uri):
    return uri.split("/")[-1].split("#")[-1].replace("_", " ")


def remove_brackets(uri):
    if uri.startswith("<") and uri.endswith(">"):
        return uri[1:-1]
    else:
        return uri


def execute_sparql_query(query, endpoint):
    """
    Perform a SPARQL query
    :param query: The SPARQL query
    :param endpoint: The SPARQL endpoint
    :return: Output of the query
    """
    endpoint.setQuery(query)
    endpoint.setReturnFormat(JSON)
    results = endpoint.query().convert()
    return results


def uri_to_sparql_format(uri):
    if uri.startswith("http://"):
        return f"<{uri}>", get_name_from_dbpedia_uri(uri)
    elif uri[1:-1].startswith("http://"):
        return f"<{uri[1:-1]}>", get_name_from_dbpedia_uri(uri[1:-1])
    elif not uri.startswith("\"") or not uri.endswith("'"):
        return f"\"{uri}\"", get_name_from_dbpedia_uri(uri)
    else:
        return uri, uri
