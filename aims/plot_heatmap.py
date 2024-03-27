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

"""Plots heatmap of properties for protein sequences"""

import matplotlib.pyplot as plt
import numpy as np

from aims_gen_properties import load_properties, compute_properties
import seaborn as sns
from pathlib import Path

property_table = load_properties()

prop_labels = ["Polarity", "Hydrophobicity", "Hydrophilicity", "pKa", "Bulkiness", "Solvent accessibility"]
prop_values = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']

prop_value_ticks = range(len(prop_values))
prop_label_ticks = np.arange(len(prop_labels)) + 0.5

n_prop = len(prop_labels)
n_values = len(prop_values)

percentage_of_max = False


def gen_heatmap(sequences_file, title, filename):
    with open(sequences_file, "r") as f:
        sequences = f.read().splitlines()
    heatmap = np.zeros((n_prop, n_values-1))

    for seq in sequences:
        seq_prop_vector = compute_properties(seq, property_table)
        for i in range(n_prop):
            prop_val = seq_prop_vector[i+1]
            map_index = min((int)(prop_val*10), n_values-2)
            heatmap[i, map_index] += 1

    if percentage_of_max:
        heatmap = 100*heatmap / np.max(heatmap)
    heatmap = heatmap.astype(int)

    ax = sns.heatmap(heatmap, annot=True, fmt='d')

    ax.set_xticks(prop_value_ticks)
    ax.set_xticklabels(prop_values)

    ax.set_yticks(prop_label_ticks)
    ax.set_yticklabels(prop_labels, rotation=0, ha='right')

    plt.title(title + "   n = " + str(len(sequences)))
    plt.savefig(filename, bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == "__main__":

    data_dir = Path("results")
    results_dir = data_dir / "plots"

    new_sequences = "new_sequences.txt"

    gen_heatmap(data_dir / new_sequences,  'Helix propensity > 0.7', results_dir / 'heatmap_70.png')
