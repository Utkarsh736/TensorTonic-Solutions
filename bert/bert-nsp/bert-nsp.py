def create_nsp_pairs(documents: list, pair_specs: list) -> list:
    """
    Returns sentence_a, sentence_b, and is_next dictionaries in a list.
    """
    result = []
    for spec in pair_specs:
        doc_a = spec["doc_a"]
        sent_a = spec["sent_a"]
        doc_b = spec["doc_b"]
        sent_b = spec["sent_b"]
    
        sentence_a = documents[doc_a][sent_a]
        sentence_b = documents[doc_b][sent_b]

        if doc_a==doc_b and sent_b == sent_a+1:
            is_next = 1
        else: is_next = 0

        result.append({"sentence_a": sentence_a, "sentence_b": sentence_b, "is_next": is_next})

    return result