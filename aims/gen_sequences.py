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

"""Generates new alpha-helical sequences"""
import numpy as np
from pathlib import Path
from aims_gen_predict import load_aims_gen_model, gen_sequences_prop
from aims_gen_properties import load_properties, n_properties

# load AIMS-GEN model
model = load_aims_gen_model()

# load properties table of amino acids.
properties = load_properties()
print(properties)

new_sequences = []

prop_min_vector = properties.min(axis=0)
prop_max_vector = properties.max(axis=0)

# generate random property values within the min and max values
n_random_values = 100
novel_sequences_random = []
prop_values_random = np.zeros((n_random_values, n_properties))
for i in range(n_properties):
    prop_values_random[:, i] = np.random.uniform(prop_min_vector[i], prop_max_vector[i], n_random_values)
print(prop_values_random)
print(prop_values_random.shape)

# generate new sequences
min_seq_len = 20  # minimum length of new sequences
max_seq_len = 40  # maximum length
for seq_len in range(min_seq_len, max_seq_len+1):
    # generate n_random_values candidate sequqnces of each sequence length in the range.
    seq_lens = n_random_values*[seq_len]
    new_sequences += gen_sequences_prop(model, prop_values_random, seq_lens)

new_sequences = set(new_sequences)

print(new_sequences)
print(len(new_sequences))

# ensure that new sequences are not contained in the reference set
print("check uniqueness ..")

with open("data/sequences.txt", "r") as f:
    sequences = f.readlines()
ref_sequences = ":".join(sequences)

new_unique_sequences = []
for seq in new_sequences:
    if not seq in ref_sequences:
        new_unique_sequences.append(seq)

# save the new sequences into file
data_dir = Path("results")
new_sequences_file = "new_gen_sequences.txt"
with open(data_dir / new_sequences_file, "w") as f:
    for sequence in new_unique_sequences:
        f.write(sequence)
        f.write('\n')

print(new_unique_sequences)
print(len(new_unique_sequences))
