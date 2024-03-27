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

"""Plots property diagrams"""

import matplotlib.pyplot as plt
from pathlib import Path
from aims_gen_properties import compute_properties, load_properties

data_dir = Path("results")
results_dir = data_dir / "plots" / "properties"
sequences_input_file = data_dir / "new_sequences.txt"

property_table = load_properties()
n_properties = 7

prop_labels = ["Helix propensity", "Polarity (Zimmerman scale)", "Hydrophobicity (Roseman scale)",
               "Hydrophilicity (Hopp-Woods scale)", "pKa (side chain only)", "Bulkiness (Zimmerman scale)",
               "Solvent accessibility (Janin)"]

filename_prefix = "properties_"
filename_ext = ["propensity", "polarity", "hydrophobicity", "hydrophilicity", "pKa", "bulkiness",
                "solvent_accessibility"]

show_plots = True

if __name__ == "__main__":

    with open(sequences_input_file, "r") as f:
        sequences = f.read().splitlines()

    for i in range(n_properties):
        prop_values = []
        len_values = []
        for seq in sequences:
            seq_prop_vector = compute_properties(seq, property_table)
            prop_values.append(seq_prop_vector[i])
            len_values.append(len(seq))
        plt.scatter(len_values, prop_values)
        plt.xlabel('Sequence length')
        plt.ylabel(prop_labels[i])
        plt.xticks(range(20, 41, 2))
        plt.title('Helix propensity, over 0.7')
        plt.savefig(results_dir / (filename_prefix + filename_ext[i] + ".png"))
        if show_plots:
            plt.show()
        plt.close()
