import os
from tqdm import tqdm

def parse_fasta(filepath, limit=None):
    """
    Generator that parses a FASTA file and yields (header, sequence) tuples.
    
    Parameters:
    - filepath (str): Path to the FASTA file.
    - limit (int, optional): Maximum number of entries to read.
    
    Yields:
    - tuple: (header, sequence)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"FASTA file not found at: {filepath}")
        
    header = None
    sequence_parts = []
    count = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if header:
                    yield header, "".join(sequence_parts)
                    count += 1
                    if limit and count >= limit:
                        return
                header = line[1:]
                sequence_parts = []
            else:
                sequence_parts.append(line)
        
        # Yield the final entry
        if header:
            if not limit or count < limit:
                yield header, "".join(sequence_parts)

def load_all_sequences(filepath, limit=None):
    """
    Loads all FASTA entries into a list. Displays a progress bar.
    
    Parameters:
    - filepath (str): Path to the FASTA file.
    - limit (int, optional): Maximum number of entries to read.
    
    Returns:
    - list of dicts: [{'header': header, 'sequence': seq}]
    """
    entries = []
    # Count total entries roughly for progress bar if limit is not set
    # (Just an estimate or none, we can use limit for tqdm if available)
    print(f"Loading protein sequences from {filepath}...")
    for header, seq in tqdm(parse_fasta(filepath, limit), total=limit, desc="Parsing FASTA"):
        entries.append({
            'header': header,
            'sequence': seq
        })
    print(f"Successfully loaded {len(entries)} sequences.")
    return entries

if __name__ == "__main__":
    # Test data loader
    DEFAULT_FASTA = r"c:\Users\nihar\AppData\Local\Temp\eb927a48-d9a4-4137-be18-e8a3e4550ca7_uniprotkb_organism_id_9606_2026_07_13.fasta.gz.ca7\uniprotkb_organism_id_9606_2026_07_13.fasta"
    try:
        entries = load_all_sequences(DEFAULT_FASTA, limit=5)
        for i, entry in enumerate(entries):
            print(f"\nEntry {i+1}:")
            print(f"Header: {entry['header'][:100]}...")
            print(f"Sequence (Length {len(entry['sequence'])}): {entry['sequence'][:50]}...")
    except Exception as e:
        print(f"Error testing data loader: {e}")
