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

""" Checks the secondary structure alpha-helix proportion of the new sequences
"""

from pathlib import Path
from aims_prot_predict import load_aims_prot_model, predict_secondary

# reads input sequences from file
data_dir = Path("results")
sequences_file = "new_gen_sequences.txt"

with open(data_dir / sequences_file, "r") as f:
    new_sequences = f.read().splitlines()

# Predicts secondary structure of each new sequence
model = load_aims_prot_model()
results = predict_secondary(model, list(new_sequences))
print(results)

# Checks that the Alpha-helix proportion is over the given threshold value
new_alpha_helix_sequences = []
alpha_helix_threshold = 0.9

sequences_alphahelix90_file = "new_sequences_alphahelix90.txt"
with open(data_dir / sequences_alphahelix90_file, "w") as f:
    for seq, sec in zip(new_sequences, results):
        if (sec.count('H')/len(seq)) > alpha_helix_threshold:
            print(sec.count('H'), len(seq))
            new_alpha_helix_sequences.append(seq)
            f.write(seq)
            f.write(' ')
            f.write(sec)
            f.write('\n')

print(new_alpha_helix_sequences)
print(len(new_alpha_helix_sequences))
