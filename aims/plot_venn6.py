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

"""Plots Venn diagram with the six properties:
'Polarity', 'Hydrophobicity', 'Hydrophilicity','pKa', 'Bulkiness', 'Solvent accessibility'
"""

import matplotlib.pyplot as plt
from pathlib import Path
import venn
from aims_gen_properties import load_properties, compute_properties

property_table = load_properties()

data_dir = Path("results")
results_dir = data_dir / "plots"

prop_labels = ["Polarity", "Hydrophobicity", "Hydrophilicity", "pKa", "Bulkiness", "Solvent accessibility"]
prop_threshold = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
n_prop = len(prop_labels)


def gen_venn_diagram(sequences, title, filename_prefix):

    vennmap = [set() for _ in range(n_prop)]
    i = 0
    for seq in sequences:
        seq_prop_vector = compute_properties(seq, property_table)[1:]
        for j in range(n_prop):
            if seq_prop_vector[j] > prop_threshold[j]:
                vennmap[j].add(i)
        i += 1

    prop_dict = dict(zip(prop_labels, vennmap))

    print(prop_dict)

    venn.venn(prop_dict, fontsize=8)

    plt.figtext(.5, 0.97, title, fontsize=10, ha='center', va='top')
    plt.savefig(results_dir / (filename_prefix + ".png"))

    plt.show()
    plt.close()


if __name__ == "__main__":

    sequences_input_file = "new_sequences.txt"
    new_sequences_70_file = data_dir / sequences_input_file

    with open(new_sequences_70_file, "r") as f:
        sequences_70 = f.read().splitlines()

    gen_venn_diagram(sequences_70, 'Helix propensity, over 0.7', "venn_70")
