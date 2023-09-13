alpha = 0.1
beta = 0.2


def simple_evaluation(entities, linked_entities, facts, true_facts):
    num_of_entities = len(entities)
    num_of_linked_entities = min(len(linked_entities), num_of_entities)
    num_of_facts = len(facts)
    num_of_true_facts = len(true_facts)

    if num_of_entities == 0:
        frac_entities = 0
    else:
        frac_entities = num_of_linked_entities / num_of_entities

    if num_of_facts == 0:
        frac_facts = 0
    else:
        frac_facts = num_of_true_facts / num_of_facts

    score = alpha * frac_entities + (1 - alpha) * frac_facts
    return score


def complex_evaluation(text, entities, linked_entities, facts, true_facts):
    text_length = len(text.split(" "))
    num_of_entities = len(entities)
    num_of_linked_entities = len(linked_entities)
    num_of_facts = len(facts)
    num_of_true_facts = len(true_facts)
    score = (beta * (alpha * num_of_entities + (1 - alpha) * num_of_facts) +
             (1 - beta) * (alpha * num_of_linked_entities + (1 - alpha) * num_of_true_facts)) / text_length
    return score
