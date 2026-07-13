def sequence_to_kmers(sequence, k=3, stride=1):
    """
    Splits a protein sequence into k-mers.
    
    Parameters:
    - sequence (str): The amino acid sequence.
    - k (int): The length of each k-mer.
    - stride (int): Stride for extraction. 
                   stride=1 for overlapping k-mers.
                   stride=k for non-overlapping k-mers.
                   
    Returns:
    - list of str: List of k-mer "words".
    """
    # Clean sequence (ensure uppercase and strip any whitespace)
    sequence = sequence.upper().strip()
    
    # If sequence is shorter than k, return it as a single k-mer (or empty list)
    if len(sequence) < k:
        return [sequence] if len(sequence) > 0 else []
        
    kmers = []
    for i in range(0, len(sequence) - k + 1, stride):
        kmers.append(sequence[i:i+k])
    return kmers

def tokenize_corpus(sequences, k=3, stride=1):
    """
    Tokenizes a list of protein sequences into a list of sentences (list of list of k-mers).
    
    Parameters:
    - sequences (list of str): List of amino acid sequences.
    - k (int): The length of each k-mer.
    - stride (int): Stride for extraction.
    
    Returns:
    - list of list of str: Tokenized corpus ready for Word2Vec training.
    """
    tokenized = []
    for seq in sequences:
        kmers = sequence_to_kmers(seq, k=k, stride=stride)
        if kmers:
            tokenized.append(kmers)
    return tokenized

if __name__ == "__main__":
    test_seq = "MTAIIKEIVSRN"
    print(f"Original Sequence: {test_seq}")
    print(f"Overlapping 3-mers (stride=1): {sequence_to_kmers(test_seq, k=3, stride=1)}")
    print(f"Non-overlapping 3-mers (stride=3): {sequence_to_kmers(test_seq, k=3, stride=3)}")
    print(f"Overlapping 4-mers (stride=1): {sequence_to_kmers(test_seq, k=4, stride=1)}")
