def wordpiece_tokenize(text: str, vocab: dict,
                       unk_token: str = "[UNK]", max_word_length: int = 100) -> list:
    """
    Returns the WordPiece tokens as a list of strings.
    """
    tokens = []

    words = text.lower().split()

    for word in words:
        if len(word)>max_word_length:
            tokens.append(unk_token)
            continue

        is_bad = False
        start = 0
        words_token = []

        while start<len(word):
            end = len(word)
            cur_substr = None

            while start<end:
                substr = word[start:end]

                if start>0:
                    substr = "##"+substr

                if substr in vocab:
                    cur_substr = substr
                    break
                end -= 1

            if cur_substr is None:
                is_bad = True
                break

            words_token.append(cur_substr)
            start=end

        if is_bad:
            tokens.append(unk_token)
        else:
            tokens.extend(words_token)
        
    return tokens