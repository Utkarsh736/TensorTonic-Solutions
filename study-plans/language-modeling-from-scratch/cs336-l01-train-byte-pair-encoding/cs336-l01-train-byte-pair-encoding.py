def apply_merge(seq, left, right, new_id): 
      out = []
      i = 0
      while i < len(seq):
        if i < len(seq) - 1 and seq[i] == left and seq[i+1] == right:
          out.append(new_id)
          i += 2
        else:
          out.append(seq[i])
          i += 1
    
      return out

def train_bpe(corpus: list[str], vocab_size: int) -> dict:
    """
    Returns a dictionary of learned vocab entries and ordered merges.
    """
    id_to_bytes = {i: bytes([i]) for i in range(256)}

    enc_corpus = []
    for s in corpus:
      enc_corpus.append( list(s.encode("utf-8")))
    
    merges = []
    while len(merges) < vocab_size - 256:
      pair_counts = {}
      for seq in enc_corpus:
        for i in range(len(seq)-1):
          pair = (seq[i], seq[i+1])
          pair_counts[pair] = pair_counts.get(pair, 0) + 1
    
      if not pair_counts: break
    
      top_count = max(pair_counts.values())
      tied = [p for p, c in pair_counts.items() if c == top_count]
      best = max(tied, key=lambda p: (id_to_bytes[p[0]], id_to_bytes[p[1]]))
    
      left, right = best
      new_id = len(merges) + 256
      id_to_bytes[new_id] = id_to_bytes[left] + id_to_bytes[right]
      enc_corpus = [apply_merge(s, left, right, new_id) for s in enc_corpus]
    
      merges.append([left, right, new_id])
    
    vocab = [[tid, list(id_to_bytes[tid])] for tid in range(256, 256 + len(merges))]
    
    return {"vocab": vocab, "merges": merges}
