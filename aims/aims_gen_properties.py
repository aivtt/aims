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
# See the License for the specific language g0overning permissions and
# limitations under the License.

'''
AIMS-GEN
Helper functions for handling protein property values
'''
import numpy as np
import pandas as pd
from pathlib import Path

sequence_chars = ('A','B','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','Z')
n_properties = 7

properties_file = "properties.csv"
data_dir = Path("data")

# Loads properties table from the properties cvs file
def load_properties():
    properties_df = pd.read_csv(data_dir/ properties_file, engine='python', index_col=None, header=None)
    properties_chars = properties_df.iloc[:,0]
    properties = np.zeros((len(sequence_chars), 7))
    for i in range(len(properties_chars)):
        index = sequence_chars.index(properties_chars[i])
        properties[index,:] = properties_df.iloc[i, 1:].to_numpy()
    return properties

# Calculates values of properties for the given sequence
def compute_properties(sequence, property_table):
    properties = np.zeros(n_properties)
    for s in sequence:
        index = sequence_chars.index(s)
        properties += property_table[index,:]
    return properties / len(sequence)

if __name__ == "__main__":
    # an example ...
    properties = load_properties()
    print(properties)

    property_values = compute_properties('HHHHH', properties)
    print(property_values)
