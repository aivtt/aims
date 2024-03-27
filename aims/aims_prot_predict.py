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
AIMS-PROT
Predicting protein secondary structure and Phi, Psi angles
"""

import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from keras.preprocessing.sequence import pad_sequences
from pathlib import Path

max_sequence_len = 750

np.random.seed(123)
tf.random.set_seed(1234)

sequence_chars = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
                  'V', 'W', 'Y', 'Z')
secondary_chars = ('B', 'E', 'G', 'H', 'I', 'S', 'T', '~')

sequence_encoder = preprocessing.LabelEncoder()
sequence_encoder.fit(sequence_chars)
secondary_encoder = preprocessing.LabelEncoder()
secondary_encoder.fit(secondary_chars)
n_sequence_chars = len(sequence_chars)
n_secondary_chars = len(secondary_chars)

print_model_summary = False


# Loads AIMS-PROT model from file
def load_aims_prot_model(model_file=Path("models") / "aims_prot_model.h5"):
    """Load trained model from model_file"""
    model = tf.keras.models.load_model(model_file, compile=False)
    if print_model_summary:
        model.summary()
    return model


# OneHotEncoding + zero padding of sequences
def seq_onehot_encoding(sequences):
    """Performs onehot encoding and zero padding for the given sequences. Returns an array of encoded sequences
    """
    encoded_sequences = np.zeros((len(sequences), max_sequence_len, n_sequence_chars), dtype=np.int8)
    for i in range(len(sequences)):
        sequence = sequences[i]
        for j in range(len(sequence)):
            seq_char_code = sequence[j]
            if seq_char_code > 0:
                encoded_sequences[i, j, seq_char_code - 1] = 1
    return encoded_sequences


# Predict sequence
def predict(sequence):
    """Predicts secondary structure and phi, psi angles. Prints the results."""
    sequence_len = len(sequence)
    encoded_sequence = sequence_encoder.transform(list(sequence)) + 1
    encoded_sequence = encoded_sequence.reshape((1, sequence_len))

    encoded_sequence = pad_sequences(encoded_sequence, maxlen=max_sequence_len, dtype=int, padding="post")

    encoded_sequence = seq_onehot_encoding(encoded_sequence)
    encoded_sequence = encoded_sequence.reshape((1, max_sequence_len, n_sequence_chars))

    model = load_aims_prot_model()
    res = model.predict(encoded_sequence)[0]

    secondary_code = np.fromiter((np.argmax(res[j, 4:n_secondary_chars + 4]) for j in range(sequence_len)), int)
    secondary_label = np.round(secondary_code).astype(int)[0:sequence_len]
    secondary_label[secondary_label < 0] = 0
    secondary_label[secondary_label > n_secondary_chars - 1] = n_secondary_chars - 1
    secondary_label = secondary_encoder.inverse_transform(secondary_label)

    print("-----")
    print(sequence)
    for i in range(sequence_len):
        phisin = res[i, 0]
        phicos = res[i, 1]
        psisin = res[i, 2]
        psicos = res[i, 3]
        phi = np.rad2deg(np.arctan2(phisin, phicos))
        psi = np.rad2deg(np.arctan2(psisin, psicos))
        if i == 0:
            phi = 180.0  # first phi is 180
        elif i == sequence_len - 1:
            psi = 180.0  # last psi is 180
        print(i, secondary_label[i], phi, psi)

    secondary_s = "".join(secondary_label)
    print(secondary_s)

# Predicts secondary structure and phi, psi angles for the given list of sequences
def predict_sequences(model, sequences):
    """Returns an array containing the predicted secondary structure codes and phi, psi angles of the given sequences
    """
    n_sequences = len(sequences)
    print("number of sequences=", n_sequences)

    encoded_sequence = []
    for sequence in sequences:
        encoded_sequence.append(sequence_encoder.transform(list(sequence)) + 1)
        print(sequence)

    max_sequences_in_batch = 200000
    n_batches = n_sequences % max_sequences_in_batch + 1

    results = np.zeros((n_sequences, max_sequence_len, 4))

    for n in range(n_batches):

        start_i = n * max_sequences_in_batch
        end_i = min(n_sequences, start_i + max_sequences_in_batch)
        if (end_i - start_i) < 1:
            break
        sequence_len = [len(sequence) for sequence in sequences[start_i:end_i]]
        encoded_sequences = seq_onehot_encoding(encoded_sequence[start_i:end_i])
        # print("predict ...")
        res = model.predict(encoded_sequences)
        # print("predicted!")

        for i in range(len(encoded_sequences)):

            secondary_code = np.fromiter(
                (np.argmax(res[i, j, 4:n_secondary_chars + 4]) for j in range(sequence_len[i])), int)
            secondary_label = secondary_code[0:sequence_len[i]]
            secondary_label[secondary_label < 0] = 0
            secondary_label[secondary_label > n_secondary_chars - 1] = n_secondary_chars - 1

            for j in range(sequence_len[i]):
                phisin = res[i, j, 0]
                phicos = res[i, j, 1]
                psisin = res[i, j, 2]
                psicos = res[i, j, 3]
                phi = np.rad2deg(np.arctan2(phisin, phicos))
                psi = np.rad2deg(np.arctan2(psisin, psicos))
                if j == 0:
                    phi = 180.0  # first phi is 180
                elif j == sequence_len[i] - 1:
                    psi = 180.0  # last psi is 180

                i1 = start_i + i
                results[i1, j, 0] = phi
                results[i1, j, 1] = psi
                results[i1, j, 2] = secondary_label[j] + 1

    return results


# Predicts secondary structure and phi, psi angles
def predict_sequences2(model, sequences):
    """ Returns list of secondary structures and phi and psi angles"""
    n_sequences = len(sequences)
    print("number of sequences=", n_sequences)

    encoded_sequence = []
    for sequence in sequences:
        encoded_sequence.append(sequence_encoder.transform(list(sequence)) + 1)
        print(sequence)

    max_sequences_in_batch = 200000
    n_batches = n_sequences % max_sequences_in_batch + 1

    # results = np.zeros((n_sequences, max_sequence_len, 4))
    results = []

    for n in range(n_batches):

        start_i = n * max_sequences_in_batch
        end_i = min(n_sequences, start_i + max_sequences_in_batch)
        if (end_i - start_i) < 1:
            break
        sequence_len = [len(sequence) for sequence in sequences[start_i:end_i]]
        encoded_sequences = seq_onehot_encoding(encoded_sequence[start_i:end_i])
        # print("predict ...")
        res = model.predict(encoded_sequences)
        # print("predicted!")

        for i in range(len(encoded_sequences)):

            sequence_result = []

            secondary_code = np.fromiter(
                (np.argmax(res[i, j, 4:n_secondary_chars + 4]) for j in range(sequence_len[i])), int)
            secondary_label = secondary_code[0:sequence_len[i]]
            secondary_label[secondary_label < 0] = 0
            secondary_label[secondary_label > n_secondary_chars - 1] = n_secondary_chars - 1

            for j in range(sequence_len[i]):
                phisin = res[i, j, 0]
                phicos = res[i, j, 1]
                psisin = res[i, j, 2]
                psicos = res[i, j, 3]
                phi = np.rad2deg(np.arctan2(phisin, phicos))
                psi = np.rad2deg(np.arctan2(psisin, psicos))
                if j == 0:
                    phi = 180.0  # first phi is 180
                elif j == sequence_len[i] - 1:
                    psi = 180.0  # last psi is 180

                i1 = start_i + i
                sequence_result.append([sequences[i1][j], phi, psi, secondary_chars[secondary_label[j]]])

            results.append(sequence_result)

    return results


# Predict secondary structure of given sequences
def predict_secondary(model, sequences):
    """Returns list of secondary structures"""

    n_sequences = len(sequences)
    print("number of sequences=", n_sequences)

    encoded_sequence = []
    for sequence in sequences:
        encoded_sequence.append(sequence_encoder.transform(list(sequence)) + 1)
        print(sequence)

    max_sequences_in_batch = 200000
    n_batches = n_sequences % max_sequences_in_batch + 1

    # results = np.zeros((n_sequences, max_sequence_len, 4))
    results = []

    for n in range(n_batches):

        start_i = n * max_sequences_in_batch
        end_i = min(n_sequences, start_i + max_sequences_in_batch)
        if (end_i - start_i) < 1:
            break
        sequence_len = [len(sequence) for sequence in sequences[start_i:end_i]]
        encoded_sequences = seq_onehot_encoding(encoded_sequence[start_i:end_i])
        # print("predict ...")
        res = model.predict(encoded_sequences)
        # print("predicted!")

        for i in range(len(encoded_sequences)):
            secondary_code = np.fromiter(
                (np.argmax(res[i, j, 4:n_secondary_chars + 4]) for j in range(sequence_len[i])), int)
            secondary_label = secondary_code[0:sequence_len[i]]
            secondary_label[secondary_label < 0] = 0
            secondary_label[secondary_label > n_secondary_chars - 1] = n_secondary_chars - 1
            secondary_label = secondary_encoder.inverse_transform(secondary_label)

            results.append("".join(secondary_label))

    return results


def print_sequences(sequences, seq_array):
    """
    Prints sequences and their related secondary structure and phi,psi angles

    :param sequences: list of sequences
    :param seq_array: array containing the secondary structure codes and phi,psi angles of the input sequences.
    """
    n_sequences, max_sequences, _ = seq_array.shape
    for i in range(n_sequences):
        print("---")
        print(i)
        print(sequences[i])
        secondary_structure = ''
        for j in range(max_sequences):
            phi = seq_array[i, j, 0]
            psi = seq_array[i, j, 1]
            secondary_code = np.rint(seq_array[i, j, 2]).astype(int)
            if secondary_code == 0:
                break
            secondary_label = secondary_chars[secondary_code - 1]
            print(j, secondary_label, np.around(phi, 5), np.around(psi, 5))
            secondary_structure += secondary_label
        print(secondary_structure)


if __name__ == "__main__":
    # Examples ...
    # Predict single sequences and print results
    predict("AAAPAGGGAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    predict("HPETLEKFDRVKHLKT")
    predict("ADAQGAMNKALELFRKDIAAKYKEL")

    # Predict list of sequences
    sequence_list = [
        'AAAPAGGGAAAAAAAAAAAAAAAAAAAAAAA',
        'HPETLEKFDRVKHLKT'
        'ADAQGAMNKALELFRKDIAAKYKEL',
    ]
    aims_model = load_aims_prot_model()
    result_list = predict_sequences(aims_model, sequence_list)
    # and print results
    print_sequences(sequence_list, result_list)
