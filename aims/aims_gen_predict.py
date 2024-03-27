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

import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from pathlib import Path

from aims_gen_properties import load_properties, compute_properties

np.random.seed(123)
tf.random.set_seed(1234)

# settings
max_sequence_len = 200
n_properties = 7

print_model_summary = True

sequence_chars = (
'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'Z')
secondary_chars = ('B', 'E', 'G', 'H', 'I', 'S', 'T', '~')

sequence_encoder = preprocessing.LabelEncoder()
sequence_encoder.fit(sequence_chars)
secondary_encoder = preprocessing.LabelEncoder()
secondary_encoder.fit(secondary_chars)
n_sequence_chars = len(sequence_chars)
n_secondary_chars = len(secondary_chars)


# Loads AIMS-GEN model
def load_aims_gen_model(model_file = Path("models") / "aims_gen_model.h5"):
    """
    Loads AIMS-GEN model from file

    :param model_file: filename
    :return: AIMS-GEN model
    """
    # load trained model from file
    model = tf.keras.models.load_model(model_file, compile=False)
    if print_model_summary:
        model.summary()
    return model


# One-hot encoding of sequences + zero padding
def onehot_encoding(sequences, label_encoder, n_chars):
    """
    The onehot_encoding function takes in a list of sequences and an encoder object,
    and returns an array with the one-hot encoding of each sequence.
    The first dimension is the number of sequences, the second is their length (padded to max_sequence_len),
    and the third is n_chars.

    :param sequences: Pass in the sequences of amino acids
    :param label_encoder: Encode the characters in the sequences
    :param n_chars: Determine the number of columns in the encoded_sequences array
    :return: An array of one-hot encoded sequences
    """
    encoded_sequences = np.zeros((len(sequences), max_sequence_len, n_chars), dtype=np.int8)
    for i in range(len(sequences)):
        sequence = label_encoder.transform(list(sequences[i])) + 1
        for j in range(len(sequence)):
            seq_char_code = sequence[j]
            if seq_char_code > 0:
                encoded_sequences[i, j, seq_char_code - 1] = 1
    return encoded_sequences


# Generate new sequences based on the given reference sequences (length and properties)
def gen_sequences(model, sequences):
    """
    The gen_sequences function takes a model and a list of sequences as input.
    It computes the properties for each sequence, and passes them to gen_sequences_prop.

    :param model: AIMS-GEN model
    :param sequences: list of reference sequences
    :return: Generated sequences list
    """
    n_sequences = len(sequences)
    # print(n_sequences)

    property_table = load_properties()
    property_values = np.zeros((n_sequences, n_properties))
    sequence_len = [len(sequence) for sequence in sequences]

    for i in range(len(sequences)):
        property_values[i, :] = compute_properties(sequences[i], property_table)

    return gen_sequences_prop(model, property_values, sequence_len)


def gen_sequences_prop(model, properties, sequence_lengths):
    """
    Generate sequences based on property vector and target sequence length

    :param model: AIMS-GEN model
    :param properties: list of property vectors
    :param sequence_lengths: list of length values
    :return: generated sequences
    """
    gen_sequences = []
    target_secondary_structures = [seq_len * 'H' for seq_len in sequence_lengths]
    encoded_secondary_structures = onehot_encoding(target_secondary_structures, secondary_encoder, n_secondary_chars)

    res = model.predict([properties, encoded_secondary_structures])

    for i in range(len(properties)):

        gen_seq_code = np.fromiter((np.argmax(res[i, j, :n_sequence_chars]) for j in range(sequence_lengths[i])), int)
        gen_seq_label = gen_seq_code[0:sequence_lengths[i]]
        gen_seq_label[gen_seq_label < 0] = 0
        gen_seq_label[gen_seq_label > n_sequence_chars - 1] = n_sequence_chars - 1
        gen_seq_label = sequence_encoder.inverse_transform(gen_seq_label)

        gen_seq_s = ""
        for s in gen_seq_label:
            gen_seq_s += str(s)

        gen_sequences.append(gen_seq_s)

    return gen_sequences
