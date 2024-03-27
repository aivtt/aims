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

""" Checks Helix propensity of the new sequences """

from pathlib import Path
from aims_gen_properties import load_properties, compute_properties

# reads input sequences from file
data_dir = Path("results")
sequences_file = "new_sequences_alphahelix90.txt"

with open(data_dir / sequences_file, "r") as f:
    new_sequences = f.read().splitlines()
print(new_sequences)

property_table = load_properties()

new_sequences_propensity_70 = []

new_sequences_propensity_70_file = "new_sequences_helices_propensity_70.txt"

for new_seq in new_sequences:
    seq = new_seq.split()[0]
    seq_prop_vector = compute_properties(seq, property_table)
    if seq_prop_vector[0] >= 0.7:
        new_sequences_propensity_70.append(seq)

print(len(new_sequences_propensity_70))

with open(data_dir / new_sequences_propensity_70_file, "w") as f:
    for new_seq in new_sequences_propensity_70:
        f.write(new_seq)
        f.write('\n')
