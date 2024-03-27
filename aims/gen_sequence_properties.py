# Copyright 2023 VTT Technical Research Centre of Finland Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Extracts properties associated with protein sequences and saves the results into csv file"""

import numpy as np
import csv
from pathlib import Path
from aims_gen_properties import compute_properties, load_properties


# Loads property table
property_table = load_properties()

sequences_file = "RMSD_less_than_3A.csv"
data_dir = Path("data")
results_data_dir = Path("results")

novel_sequences_70 = []

result_sequences_prop_file = "new_sequences_with_properties.csv"

# Compute property values of sequences
sequence_properties = []
sequence_ids = []
with open(results_data_dir/sequences_file, "r") as f:
    lines = f.read().splitlines()
    for line in lines:
        print(line)
        dataLine = line.split("_")
        print(dataLine)
        sequence = dataLine[3].split(",")[0]
        seq_id = dataLine[2]
        novel_sequences_70.append(sequence)
        sequence_ids.append(id)
        sequence_properties.append([int(seq_id), sequence, len(sequence)] +
                                   np.round(compute_properties(sequence, property_table), 4).tolist())

sequence_properties.sort(key=lambda seq: seq[0])
print(sequence_properties)

# write results into csv file
prop_labels = ["Helix propensity", "Polarity (Zimmerman scale)", "Hydrophobicity (Roseman scale)",
               "Hydrophilicity (Hopp-Woods scale)", "pKa (side chain only)", "Bulkiness (Zimmerman scale)",
               "Solvent accessibility (Janin)"]

with open(results_data_dir/result_sequences_prop_file, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Id", "Sequence", "Length"] + prop_labels)
    for seq_prop in sequence_properties:
        writer.writerow(seq_prop)
