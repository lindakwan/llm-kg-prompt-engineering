from SPARQLWrapper import JSON, SPARQLWrapper


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


def uri_to_sparql_format_dbpedia(uri):
    if uri.startswith("http://"):
        return f"<{uri}>", get_name_from_dbpedia_uri(uri)
    elif uri[1:-1].startswith("http://"):
        return f"<{uri[1:-1]}>", get_name_from_dbpedia_uri(uri[1:-1])
    elif not uri.startswith("\"") or not uri.endswith("'"):
        return f"\"{uri}\"", get_name_from_dbpedia_uri(uri)
    else:
        return uri, uri


def uri_to_sparql_format_wikidata(uri):
    if uri.startswith("http://"):
        return f"<{uri}>"
    else:
        return uri


def get_sparql_results_wikidata(s_uri, p_uri, o_uri):
    sparql_wd = SPARQLWrapper("https://query.wikidata.org/sparql")

    # Convert the triple to SPARQL format
    s_format = uri_to_sparql_format_wikidata(s_uri)
    p_format = uri_to_sparql_format_wikidata(p_uri)
    o_format = uri_to_sparql_format_wikidata(o_uri)

    sparql_query_p = f"""
    SELECT ?predicate ?propLabel WHERE {{
        {s_format} ?predicate {o_format}.
        SERVICE wikibase:label {{
            bd:serviceParam wikibase:language "en" . }}
        ?prop wikibase:directClaim ?predicate .
        ?prop rdfs:label ?propLabel.  filter(lang(?propLabel) = "en"). }}"""

    sparql_query_o = f"""
    SELECT ?object ?objectLabel WHERE {{
        {s_format} {p_format} ?object.
        SERVICE wikibase:label {{
            bd:serviceParam wikibase:language "en" . }}
        FILTER(!isLiteral(?object) || lang(?object) = "" || langMatches(lang(?object), "EN")) . }}"""

    # Perform the SPARQL query
    sparql_result_p = execute_sparql_query(sparql_query_p, sparql_wd)
    results_bindings_p = sparql_result_p["results"]["bindings"]
    print(sparql_query_p)

    pred_uri_label_pairs = [(result_bind["predicate"]["value"], result_bind["propLabel"]["value"])
                            for result_bind in results_bindings_p]

    sparql_result_o = execute_sparql_query(sparql_query_o, sparql_wd)
    results_bindings_o = sparql_result_o["results"]["bindings"]
    print(sparql_query_o)

    obj_uri_label_pairs = [(result_bind["object"]["value"], result_bind["objectLabel"]["value"])
                           for result_bind in results_bindings_o]

    return pred_uri_label_pairs, obj_uri_label_pairs


def get_sparql_results_wikidata_described(s_uri, p_uri, o_uri):
    sparql_wd = SPARQLWrapper("https://query.wikidata.org/sparql")

    # Convert the triple to SPARQL format
    s_format = uri_to_sparql_format_wikidata(s_uri)
    p_format = uri_to_sparql_format_wikidata(p_uri)
    o_format = uri_to_sparql_format_wikidata(o_uri)

    sparql_query_p1 = f"""
    SELECT ?predicate ?propLabel ?propDescription WHERE {{
        {s_format} ?predicate {o_format}.
        SERVICE wikibase:label {{
            bd:serviceParam wikibase:language "en" . }}
        ?prop wikibase:directClaim ?predicate .
        ?prop rdfs:label ?propLabel.  filter(lang(?propLabel) = "en"). }}"""

    sparql_query_p2 = f"""
        SELECT ?predicate ?propLabel ?propDescription WHERE {{
            {o_format} ?predicate {s_format}.
            SERVICE wikibase:label {{
                bd:serviceParam wikibase:language "en" . }}
            ?prop wikibase:directClaim ?predicate .
            ?prop rdfs:label ?propLabel.  filter(lang(?propLabel) = "en"). }}"""

    sparql_query_o = f"""
    SELECT ?object ?objectLabel ?objectDescription WHERE {{
        {s_format} {p_format} ?object.
        SERVICE wikibase:label {{
            bd:serviceParam wikibase:language "en" . }}
        FILTER(!isLiteral(?object) || lang(?object) = "" || langMatches(lang(?object), "EN")) . }}"""

    # Perform the SPARQL query
    sparql_result_p1 = execute_sparql_query(sparql_query_p1, sparql_wd)
    results_bindings_p1 = sparql_result_p1["results"]["bindings"]
    print(sparql_query_p1)

    # Create a list of tuples containing the predicate URI, label, and description
    pred_uri_label_trips_forw = [(result_bind["predicate"]["value"], result_bind["propLabel"]["value"],
                                  result_bind["propDescription"]["value"] if "propDescription" in result_bind else None
                                  ) for result_bind in results_bindings_p1]

    # Perform the SPARQL query
    sparql_result_p2 = execute_sparql_query(sparql_query_p2, sparql_wd)
    results_bindings_p2 = sparql_result_p2["results"]["bindings"]
    print(sparql_query_p2)

    pred_uri_label_trips_back = [(result_bind["predicate"]["value"], result_bind["propLabel"]["value"],
                                  result_bind["propDescription"]["value"] if "propDescription" in result_bind else None
                                  ) for result_bind in results_bindings_p2]

    sparql_result_o = execute_sparql_query(sparql_query_o, sparql_wd)
    results_bindings_o = sparql_result_o["results"]["bindings"]
    print(sparql_query_o)

    # Create a list of tuples containing the object URI, label, and description
    obj_uri_label_trips = [(result_bind["object"]["value"], result_bind["objectLabel"]["value"],
                            result_bind["objectDescription"]["value"] if "objectDescription" in result_bind else None
                            ) for result_bind in results_bindings_o]

    return pred_uri_label_trips_forw, pred_uri_label_trips_back, obj_uri_label_trips
