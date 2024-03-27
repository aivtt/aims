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

"""
AIMS test case
"""
print(__doc__)

import sys

sys.path.append('aims')

create_plots = False # change this True to create plots (heatmap, venn diagrams, properties)

# Generates candidate Alpha-helix sequences
with open("aims/gen_sequences.py") as f:
    exec(f.read())

# Check Alpha-helix proportion in secondary structures of the generated sequences
with open("aims/gen_check_alpha_helix.py") as f:
    exec(f.read())

# Ensures that helices propensity is below the set threshold value (70%)
with open("aims/gen_check_helices_propensity.py") as f:
    exec(f.read())

# Creates a cvs file that summaries the properties of the resulting candidate sequences
with open("aims/gen_sequence_properties.py") as f:
    exec(f.read())

if create_plots:
    with open("aims/plot_heatmap.py") as f:
        exec(f.read())

    with open("aims/plot_venn3.py") as f:
        exec(f.read())

    with open("aims/plot_venn6.py") as f:
        exec(f.read())

    with open("aims/plot_properties.py") as f:
        exec(f.read())