import pandas as pd
import numpy as np

def generate_position_maps(sequences):
    # Determine the length of the sequences
    sequence_length = len(sequences[0])
    # Initialize a list to hold the set of amino acids per position
    position_aas = [set() for _ in range(sequence_length)]

    for seq in sequences:
        for index, amino_acid in enumerate(seq):
            position_aas[index].add(amino_acid)

    # Create a mapping from amino acid to binary code for each position
    position_maps = []
    for aas in position_aas:
        # Convert set to list and sort to ensure consistent ordering
        aas = sorted(list(aas))
        # Map each amino acid to an index
        map = {amino_acid: idx for idx, amino_acid in enumerate(aas)}
        position_maps.append(map)
    
    return position_maps

def encode_sequences_with_maps(sequences, position_maps):
    encoded_sequences = []
    for seq in sequences:
        encoded_seq = [position_maps[i][seq[i]] for i in range(len(seq))]
        encoded_sequences.append(encoded_seq)
    
    return np.array(encoded_sequences)

# Example DataFrame and sequence column
data = {'mutant_sequence': ['ACGT', 'TGCA', 'ATGC', 'CGTA']}
df = pd.DataFrame(data)
sequences = df['mutant_sequence'].tolist()

# Generate position-specific amino acid maps
position_maps = generate_position_maps(sequences)

# Encode the sequences using the generated maps
encoded_sequences = encode_sequences_with_maps(sequences, position_maps)
print(encoded_sequences)
