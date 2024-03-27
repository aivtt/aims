# Data files

## properties.csv

- Protein property table containing normalized property values for each amino acid
- Columns: Amino acid code, Helix propensity, Polarity, Hydrophobicity, Hydrophilicity, pKa, Bulkiness, Solvent accessibility.
  
## sequences.txt

- All sequences. Minimum sequence length is 10.
  
## sequence_ids.txt

- The four character id codes of the proteins used in the training of the models. The corresponding pdb files of the proteins are available from <https://www.rcsb.org/>.

## sequence_ids_750.txt

- The max 750 length sequences from the full set sequence_ids.txt
- used in training the AIMS-PROT model.
   
## sequence_ids_200.txt

- The max 200 length sequences from the full set sequence_ids.txt
- used in training the AIMS-GENERATE model.
    
