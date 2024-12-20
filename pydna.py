import sys
import os
import traceback
import datetime
import time
import logging
import configparser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
import tensorflow as tf

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from zope.interface import implementer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import (
    Conv1D,
    LSTM,
    Dense,
    Dropout,
    Masking,
    Embedding,
    Flatten,
    MaxPooling1D,
)
from tensorflow.keras.models import load_model
import pickle as pkl

# from keras.utils import to_categorical
import re

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# How to write Getter/Setter for static variables
# https://samwize.com/2012/09/20/how-to-write-getter-slash-setter-for-static-variables/


class PyDNA(object):
    """main class pydna library

    args:
        object (_type_): _description_

    returns:
        _type_: _description_
    """

    _app_config_file = None
    _app_log_file = None
    _app_is_log = False

    @staticmethod
    def is_dna(
        dna_sequence_string, base_sequence=["A", "C", "G", "T"], minimum_percentage=100
    ):
        total_dna_base = 0
        try:
            dna_sequence_string = dna_sequence_string.upper()
            for base in base_sequence:
                total_dna_base = total_dna_base + dna_sequence_string.count(
                    base.upper()
                )
            dna_fraction = total_dna_base / len(dna_sequence_string)
            dna_fraction = dna_fraction * 100
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return True if dna_fraction >= minimum_percentage else False

    @staticmethod
    def dna_count_nucleotide(
        dna_sequence_string, base_sequence=["A", "C", "G", "T"], is_length=None
    ):
        dna_sequence_length = 0
        base_sequence_count = {}
        try:
            dna_sequence_string = dna_sequence_string.upper()
            for base in base_sequence:
                base_count = dna_sequence_string.count(base.upper())
                base_sequence_count[base] = base_count
                if is_length is not None:
                    dna_sequence_length += base_count
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return base_sequence_count, dna_sequence_length

    @staticmethod
    def rna_count_nucleotide(
        rna_sequence_string, base_sequence=["A", "C", "G", "U"], is_length=None
    ):
        rna_sequence_length = 0
        base_sequence_count = {}
        try:
            rna_sequence_string = rna_sequence_string.upper()
            for base in base_sequence:
                base_count = rna_sequence_string.count(base.upper())
                base_sequence_count[base] = base_count
                if is_length is not None:
                    rna_sequence_length += base_count
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return base_sequence_count, rna_sequence_length

    @staticmethod
    def dna_sequence_reverse(dna_sequence_string):
        try:
            dna_reverse_sequence = "".join(reversed(dna_sequence_string.upper()))
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return dna_reverse_sequence

    @staticmethod
    def dna_sequence_complement(dna_sequence_string):
        try:
            complement_pairs = {
                "A": "T",
                "T": "A",
                "G": "C",
                "C": "G",
                "U": "A",
                "CT": "R",
                "AG": "Y",
                "GC": "S*",
                "AT": "W*",
                "T/UG": "M",
                "AC": "K",
                "CGT": "V",
                "AGT": "H",
                "ACT": "D",
                "ACG": "B",
                "ACGT": "N",
            }
            dna_complement_sequence = "".join(
                [complement_pairs[base] for base in dna_sequence_string.upper()]
            )
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return dna_complement_sequence

    @staticmethod
    def dna_sequence_reverse_complement(dna_sequence_string):
        try:
            reverse_complement_pairs = {
                "A": "T",
                "T": "A",
                "G": "C",
                "C": "G",
                "U": "A",
                "CT": "R",
                "AG": "Y",
                "GC": "S*",
                "AT": "W*",
                "T/UG": "M",
                "AC": "K",
                "CGT": "V",
                "AGT": "H",
                "ACT": "D",
                "ACG": "B",
                "ACGT": "N",
            }
            dna_reverse_complement_sequence = "".join(
                [
                    reverse_complement_pairs.get(base, base)
                    for base in reversed(dna_sequence_string.upper())
                ]
            )
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return dna_reverse_complement_sequence

    @staticmethod
    def dna_count_gc_content(dna_sequence_string):
        dna_sequence_length = None
        try:
            a_nucleotide = dna_sequence_string.count("A")
            c_nucleotide = dna_sequence_string.count("C")
            g_nucleotide = dna_sequence_string.count("G")
            t_nucleotide = dna_sequence_string.count("T")
            a_c_g_t_count = a_nucleotide + c_nucleotide + g_nucleotide + t_nucleotide
            g_c_count = g_nucleotide + c_nucleotide
            dna_gc_content = (g_c_count / a_c_g_t_count) * 100
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return dna_gc_content

    @staticmethod
    def is_rna(
        rna_sequence_string, base_sequence=["A", "C", "G", "U"], minimum_percentage=100
    ):
        total_rna_base = 0
        try:
            rna_sequence_string = rna_sequence_string.upper()
            for base in base_sequence:
                total_rna_base = total_rna_base + rna_sequence_string.count(
                    base.upper()
                )
            rna_fraction = total_rna_base / len(rna_sequence_string)
            rna_fraction = rna_fraction * 100
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return True if rna_fraction >= minimum_percentage else False

    @staticmethod
    def dna_sequence_np_array(dna_sequence_string):
        try:
            dna_sequence_string = dna_sequence_string.lower()
            regex_acgt = re.compile("[^acgt]")
            if regex_acgt.search(dna_sequence_string) == None:
                dna_sequence_array = np.array(list(dna_sequence_string))
            else:
                dna_sequence_array = None
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return dna_sequence_array

    @staticmethod
    def dna_sequence_pattern(dna_sequence_string, dna_sequence_pattern):
        search_result = False
        try:
            search_pattern = re.search(
                dna_sequence_pattern.lower(), dna_sequence_string.lower()
            )
            if search_pattern:
                search_result = True
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return search_result

    # @staticmethod
    # def k_mers_words_original(dna_sequence_string, k_mers_length=6):
    #     try:
    #         k_mers_list = [dna_sequence_string[x:x + k_mers_length].lower() for x in range(len(dna_sequence_string) - k_mers_length + 1)]
    #     except:
    #         print(PyDNA.get_exception_info())
    #         if PyDNA._app_is_log: PyDNA.write_log_file("error", PyDNA.get_exception_info())
    #     return k_mers_list

    @staticmethod
    def k_mer_words_original(dna_sequence_string, k_mer_length=6):
        try:
            k_mer_list = [
                dna_sequence_string[x : x + k_mer_length].lower()
                for x in range(len(dna_sequence_string) - k_mer_length + 1)
            ]
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return k_mer_list

    @staticmethod
    def k_mer_words(dna_sequence_string, k_mer_length=6):
        try:
            k_mer_list = []
            dna_sequence_string = dna_sequence_string.lower()
            length = len(dna_sequence_string) - k_mer_length + 1
            for x in range(length):
                k_mer_list.append(dna_sequence_string[x : x + k_mer_length])
            k_mer_numpy_array = np.array(k_mer_list)
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return k_mer_list, k_mer_numpy_array

    @staticmethod
    def dna_label_encoder(dna_sequence_array):
        try:
            label_encoder = LabelEncoder()
            label_encoder.fit(np.array(["a", "c", "g", "t"]))
            label_encoder_integer = label_encoder.transform(dna_sequence_array)
            label_encoder_float = label_encoder_integer.astype(np.float32)
            label_encoder_float[label_encoder_float == 0] = 0.25
            label_encoder_float[label_encoder_float == 1] = 0.50
            label_encoder_float[label_encoder_float == 2] = 0.75
            label_encoder_float[label_encoder_float == 3] = 1.00
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return label_encoder_float

    @staticmethod
    def dna_onehot_encoder(dna_sequence_array):
        try:
            label_encoder = LabelEncoder()
            label_encoder.fit(np.array(["a", "c", "g", "t"]))
            int_encoded = label_encoder.transform(dna_sequence_array)
            onehot_encoder = OneHotEncoder(sparse=False)
            int_encoded = int_encoded.reshape(len(int_encoded), 1)
            onehot_encoded = onehot_encoder.fit_transform(int_encoded)
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return onehot_encoded

    @staticmethod
    def data_frame_label_encoder(data_frame_serie):
        try:
            label_encoder = LabelEncoder()
            data_frame_serie = label_encoder.fit_transform(data_frame_serie)
            label_encoder_class = label_encoder.classes_
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return data_frame_serie, label_encoder_class

    @staticmethod
    def data_frame_flatten_array(data_frame_serie):
        try:
            data_frame_serie = np.ravel(data_frame_serie)
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return data_frame_serie

    @staticmethod
    def data_frame_astype(data_frame_serie, data_type):
        try:
            data_frame_serie = data_frame_serie.astype(data_type)
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return data_frame_serie

    @staticmethod
    def balance_class_smote(X, y):
        try:
            smote_over_sampling = SMOTE(random_state=50, n_jobs=-1)
            X, y = smote_over_sampling.fit_resample(X, y)
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return X, y

    @staticmethod
    def X_feature_standard_scaler(standard_scaler, X):
        try:
            standard_scaler.fit(X)
            X = standard_scaler.transform(X)
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return X

    @staticmethod
    def X_pca(pca, X):
        try:
            X = pca.fit_transform(X)
            pca_explained_variance = pca.explained_variance_ratio_
            cumulative_sum_eigenvalues = np.cumsum(pca_explained_variance)
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return X, pca_explained_variance, cumulative_sum_eigenvalues

    @staticmethod
    def X_train_valid_pca(pca, X_train, X_valid):
        try:
            X_train = pca.fit_transform(X_train)
            X_valid = pca.transform(X_valid)
            pca_explained_variance = pca.explained_variance_ratio_
            cumulative_sum_eigenvalues = np.cumsum(pca_explained_variance)
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return X_train, X_valid, pca_explained_variance, cumulative_sum_eigenvalues

    @staticmethod
    def pca_explained_variance_plot(
        pca_n_components, explained_variance, min_explained_variance, font_size
    ):
        plt.figure(figsize=(5, 4))
        plt.style.use("ggplot")
        plt.tick_params(labelsize=font_size)
        rects = plt.bar(pca_n_components, explained_variance, alpha=0.5, align="center")
        for rect in rects:
            rec_initial = rect.get_x()
            rec_width = rect.get_width()
            rec_height = rect.get_height()
            plt.text(
                rec_initial + rec_width / 2,
                rec_height - min_explained_variance,
                str(float("{0:.3f}".format(rec_height))),
                horizontalalignment="center",
                verticalalignment="bottom",
                fontsize=font_size,
            )
        plt.ylabel("Explained Variance Ratio", fontsize=font_size)
        plt.xlabel("Principal Components", fontsize=font_size)
        plt.title("PCA Individual Explained Variance", fontsize=font_size)
        plt.show()

    @staticmethod
    def pca_component_accuracy_scores_plot(
        pca_n_components, accuracy_score_valid, accuracy_score_test
    ):
        font_size = 8
        title_label = "Classification Validation and Test Accuracy Scores"
        x_label = "Principal Components"
        y_label = "Accuracy Score, %"
        legend_label = ["Validation", "Test"]
        plt.figure(figsize=[5, 5])
        plt.style.use("ggplot")
        plt.tick_params(labelsize=font_size)
        plt.plot(pca_n_components, accuracy_score_valid)
        plt.plot(pca_n_components, accuracy_score_test)
        plt.xlabel(x_label, fontsize=font_size)
        plt.ylabel(y_label, fontsize=font_size)
        plt.legend(legend_label, fontsize=font_size)
        plt.title(title_label, fontsize=font_size)
        plt.show()

    # @staticmethod
    # def dna_onehot_encoder_keras(dna_sequence_array):
    #     try:
    #         label_encoder = LabelEncoder()
    #         label_encoder.fit(np.array(['a','c','g','t']))
    #         label_encoder_integer = label_encoder.transform(dna_sequence_array)
    #         onehot_encoder = to_categorical(label_encoder_integer)
    #     except:
    #         print(PyDNA.get_exception_info())
    #         if PyDNA._app_is_log: PyDNA.write_log_file("error", PyDNA.get_exception_info())
    #     return onehot_encoder

    @staticmethod
    def bag_of_word_list(word_list, word_ngram):
        matrix_token_counts = None
        try:
            count_vectorizer = CountVectorizer(ngram_range=(word_ngram, word_ngram))
            matrix_token_counts = count_vectorizer.fit_transform(word_list)
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return matrix_token_counts

    @staticmethod
    def bag_of_word_series(word_series, word_ngram):
        """
        convert a collection of text documents to a matrix of token counts
        """
        X = None
        try:
            texts_list = list(word_series)
            for item in range(len(texts_list)):
                texts_list[item] = " ".join(texts_list[item])
            count_vectorizer = CountVectorizer(ngram_range=(word_ngram, word_ngram))
            X = count_vectorizer.fit_transform(texts_list)
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return X

    @staticmethod
    def create_dataframe_column_words(
        dna_data_frame, input_column_name, output_column_name
    ):
        try:
            dna_data_frame[output_column_name] = dna_data_frame.apply(
                lambda x: PyDNA.k_mer_words_original(x[input_column_name]), axis=1
            )
            dna_data_frame = dna_data_frame.drop(input_column_name, axis=1)
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return dna_data_frame

    @staticmethod
    def dna_count_nucleotide(dna_sequence_string, is_length=None):
        dna_sequence_length = None
        try:
            a_nucleotide = dna_sequence_string.count("A")
            c_nucleotide = dna_sequence_string.count("C")
            g_nucleotide = dna_sequence_string.count("G")
            t_nucleotide = dna_sequence_string.count("T")
            if is_length is not None:
                dna_sequence_length = len(dna_sequence_string)
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return (
            a_nucleotide,
            c_nucleotide,
            g_nucleotide,
            t_nucleotide,
            dna_sequence_length,
        )

    @staticmethod
    def rna_count_nucleotide(rna_sequence_string, is_length=None):
        rna_sequence_length = None
        try:
            a_nucleotide = rna_sequence_string.count("A")
            c_nucleotide = rna_sequence_string.count("C")
            g_nucleotide = rna_sequence_string.count("G")
            u_nucleotide = rna_sequence_string.count("U")
            if is_length is not None:
                rna_sequence_length = len(rna_sequence_string)
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return (
            a_nucleotide,
            c_nucleotide,
            g_nucleotide,
            u_nucleotide,
            rna_sequence_length,
        )

    # Reverse Complement converts a DNA sequence into its reverse, complement, or reverse-complement counterpart.
    # You may want to work with the reverse-complement of a sequence if it contains an ORF on the reverse strand.

    @staticmethod
    def dna_protein_translation(dna_sequence_string):
        protein_sequence = str()
        try:
            dna_codon_table = {
                "TTT": "F",
                "CTT": "L",
                "ATT": "I",
                "GTT": "V",
                "TTC": "F",
                "CTC": "L",
                "ATC": "I",
                "GTC": "V",
                "TTA": "L",
                "CTA": "L",
                "ATA": "I",
                "GTA": "V",
                "TTG": "L",
                "CTG": "L",
                "ATG": "M",
                "GTG": "V",
                "TCT": "S",
                "CCT": "P",
                "ACT": "T",
                "GCT": "A",
                "TCC": "S",
                "CCC": "P",
                "ACC": "T",
                "GCC": "A",
                "TCA": "S",
                "CCA": "P",
                "ACA": "T",
                "GCA": "A",
                "TCG": "S",
                "CCG": "P",
                "ACG": "T",
                "GCG": "A",
                "TAT": "Y",
                "CAT": "H",
                "AAT": "N",
                "GAT": "D",
                "TAC": "Y",
                "CAC": "H",
                "AAC": "N",
                "GAC": "D",
                "TAA": "STOP",
                "CAA": "Q",
                "AAA": "K",
                "GAA": "E",
                "TAG": "STOP",
                "CAG": "Q",
                "AAG": "K",
                "GAG": "E",
                "TGT": "C",
                "CGT": "R",
                "AGT": "S",
                "GGT": "G",
                "TGC": "C",
                "CGC": "R",
                "AGC": "S",
                "GGC": "G",
                "TGA": "STOP",
                "CGA": "R",
                "AGA": "R",
                "GGA": "G",
                "TGG": "W",
                "CGG": "R",
                "AGG": "R",
                "GGG": "G",
            }
            dna_length = len(dna_sequence_string)
            for i in range(0, dna_length - (3 + dna_length % 3), 3):
                if dna_codon_table[dna_sequence_string[i : i + 3]] == "STOP":
                    break
                protein_sequence += dna_codon_table[dna_sequence_string[i : i + 3]]
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return protein_sequence

    @staticmethod
    def dna_rna_transcription(dna_sequence_string):
        try:
            rna_sequence = dna_sequence_string.replace("T", "U")
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return rna_sequence

    @staticmethod
    def rna_protein_translation(rna_sequence_string):
        rna_protein_sequence = str()
        try:
            rna_codon_table = {
                "UUU": "F",
                "CUU": "L",
                "AUU": "I",
                "GUU": "V",
                "UUC": "F",
                "CUC": "L",
                "AUC": "I",
                "GUC": "V",
                "UUA": "L",
                "CUA": "L",
                "AUA": "I",
                "GUA": "V",
                "UUG": "L",
                "CUG": "L",
                "AUG": "M",
                "GUG": "V",
                "UCU": "S",
                "CCU": "P",
                "ACU": "T",
                "GCU": "A",
                "UCC": "S",
                "CCC": "P",
                "ACC": "T",
                "GCC": "A",
                "UCA": "S",
                "CCA": "P",
                "ACA": "T",
                "GCA": "A",
                "UCG": "S",
                "CCG": "P",
                "ACG": "T",
                "GCG": "A",
                "UAU": "Y",
                "CAU": "H",
                "AAU": "N",
                "GAU": "D",
                "UAC": "Y",
                "CAC": "H",
                "AAC": "N",
                "GAC": "D",
                "UAA": "STOP",
                "CAA": "Q",
                "AAA": "K",
                "GAA": "E",
                "UAG": "STOP",
                "CAG": "Q",
                "AAG": "K",
                "GAG": "E",
                "UGU": "C",
                "CGU": "R",
                "AGU": "S",
                "GGU": "G",
                "UGC": "C",
                "CGC": "R",
                "AGC": "S",
                "GGC": "G",
                "UGA": "STOP",
                "CGA": "R",
                "AGA": "R",
                "GGA": "G",
                "UGG": "W",
                "CGG": "R",
                "AGG": "R",
                "GGG": "G",
            }
            rna_length = len(rna_sequence_string)
            for i in range(0, rna_length - (3 + rna_length % 3), 3):
                if rna_codon_table[rna_sequence_string[i : i + 3]] == "STOP":
                    break
                rna_protein_sequence += rna_codon_table[rna_sequence_string[i : i + 3]]
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return rna_protein_sequence

    @staticmethod
    def imbalanced_classes_plot(
        ds_y, is_class_number, class_column_name, x_label, y_label, title, font_size=8
    ):
        try:
            plt.figure(figsize=(5, 5))
            sns.set(style="darkgrid")
            if is_class_number == True:
                ax = sns.countplot(x=ds_y)
            else:
                ax = sns.countplot(x=class_column_name, data=ds_y)
            ax.set_xlabel(x_label, fontsize=font_size)
            ax.set_ylabel(y_label, fontsize=font_size)
            ax.tick_params(labelsize=font_size)
            for p in ax.patches:
                height = p.get_height()
                ax.text(
                    p.get_x() + p.get_width() / 2.0,
                    height + 10,
                    "{:1.0f}".format(height),
                    ha="center",
                    fontsize=font_size,
                )
            plt.title(title, fontsize=font_size)
            plt.show()
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())

    @staticmethod
    def pandas_read_data(file_type, file_path_name, column_name=None):
        try:
            if file_type == "CSV":
                if column_name is None:
                    df_read = pd.read_csv(filepath_or_buffer=file_path_name)
                else:
                    df_read = pd.read_csv(
                        filepath_or_buffer=file_path_name, names=column_name
                    )
            elif file_type == "TXT":
                if column_name is None:
                    df_read = pd.read_table(filepath_or_buffer=file_path_name)
                else:
                    df_read = pd.read_table(
                        filepath_or_buffer=file_path_name, names=column_name
                    )
            elif file_type == "JSON":
                pass
            elif file_type == "HTML":
                pass
            elif file_type == "EXCEL":
                pass
            elif file_type == "HDF5":
                pass
            elif file_type == "PICKLE":
                pass
            elif file_type == "SQL":
                pass
            else:
                pass
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return df_read

    @staticmethod
    def select_df_column(data_frame, column_name):
        try:
            column_series = data_frame[column_name]
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return column_series

    @staticmethod
    def select_y_label(data_frame, column_name):
        try:
            y = data_frame[column_name]
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return y

    @staticmethod
    def select_X_feature(data_frame, column_name):
        try:
            X = data_frame.drop(labels=column_name, axis=1)
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return X

    @staticmethod
    def train_validation_test_split(
        X, y, is_stratify=True, test_size=0.2, valid_size=0.5, random_state_seed=50
    ):
        try:
            if is_stratify == True:
                X_train, X_test_valid, y_train, y_test_valid = train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    stratify=y,
                    random_state=random_state_seed,
                )
                X_valid, X_test, y_valid, y_test = train_test_split(
                    X_test_valid,
                    y_test_valid,
                    test_size=valid_size,
                    stratify=y_test_valid,
                    random_state=random_state_seed,
                )
            else:
                X_train, X_test_valid, y_train, y_test_valid = train_test_split(
                    X, y, test_size=test_size, random_state=random_state_seed
                )
                X_valid, X_test, y_valid, y_test = train_test_split(
                    X_test_valid,
                    y_test_valid,
                    test_size=valid_size,
                    random_state=random_state_seed,
                )
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return X_train, y_train, X_valid, y_valid, X_test, y_test

    @staticmethod
    def create_ml_model(model_name, X_feature_train, y_label_train):
        try:
            if model_name == "MultinomialNB":
                # ml_model = MultinomialNB() #t this need to be tested
                ml_model = Pipeline(
                    [
                        ("Normalizing", MinMaxScaler()),
                        ("MultinomialNB", MultinomialNB()),
                    ]
                )
            elif model_name == "GaussianNB":
                ml_model = GaussianNB()
            elif model_name == "MLPClassifier":
                ml_model = MLPClassifier(random_state=50)
            elif model_name == "RandomForestClassifier":
                ml_model = RandomForestClassifier(n_jobs=-1, random_state=50)
            elif model_name == "XGBClassifier":
                ml_model = XGBClassifier(
                    n_jobs=-1, random_state=50, objective="binary:softmax", num_class=7
                )
            elif model_name == "LogisticRegression":
                ml_model = LogisticRegression(
                    multi_class="multinomial",
                    solver="newton-cg",
                    n_jobs=-1,
                    random_state=50,
                )
            elif model_name == "DecisionTreeClassifier":
                ml_model = DecisionTreeClassifier(random_state=50)
            elif model_name == "GradientBoostingClassifier":
                ml_model = GradientBoostingClassifier(random_state=50)
            elif model_name == "AdaBoostClassifier":
                ml_model = AdaBoostClassifier(random_state=50)
            elif model_name == "HistGradientBoostingClassifier":
                ml_model = HistGradientBoostingClassifier(random_state=50)
            elif model_name == "SVC":
                ml_model = SVC(random_state=50, kernel="linear")
            elif model_name == "LGBMClassifier":
                ml_model = LGBMClassifier(
                    objective="multiclass", random_state=50, n_jobs=-1
                )
            elif model_name == "CatBoostClassifier":
                ml_model = CatBoostClassifier(verbose=False)
            elif model_name == "KNeighborsClassifier":
                ml_model = KNeighborsClassifier(n_jobs=-1)
            ml_model.fit(X_feature_train, y_label_train)
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return ml_model

    @staticmethod
    def create_cnn_model(
        y_train,
        X_train,
        epochs_value,
        validation_split_value,
        val_accuracy_threshold,
        verbose_value=0,
    ):
        try:
            model = Sequential()
            model.add(
                Conv1D(filters=32, kernel_size=12, input_shape=(X_train.shape[1], 4))
            )
            model.add(MaxPooling1D(pool_size=4))
            model.add(Flatten())
            model.add(Dense(units=16, activation="relu"))
            model.add(Dense(units=2, activation="softmax"))
            model.compile(
                loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
            )
            callback = ValAccuracyCallback(
                val_accuracy_threshold=val_accuracy_threshold
            )
            history = model.fit(
                X_train,
                y_train,
                epochs=epochs_value,
                verbose=verbose_value,
                validation_split=validation_split_value,
                callbacks=[callback],
                use_multiprocessing=True,
            )
            model.summary()
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return model, history

    @staticmethod
    def create_lstm_model(
        y_train,
        X_train,
        epochs_value,
        validation_split_value,
        val_accuracy_threshold,
        verbose_value=0,
    ):
        try:
            model = Sequential()
            model.add(
                LSTM(
                    64,
                    return_sequences=True,
                    dropout=0.1,
                    recurrent_dropout=0.1,
                    input_shape=(X_train.shape[1], X_train.shape[2]),
                )
            )
            model.add(MaxPooling1D(pool_size=4))
            model.add(Flatten())
            model.add(Masking(mask_value=0.0))
            model.add(Dense(64, activation="relu"))
            model.add(Dropout(0.5))
            model.add(Dense(2, activation="softmax"))
            model.compile(
                optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
            )
            callback = ValAccuracyCallback(
                val_accuracy_threshold=val_accuracy_threshold
            )
            history = model.fit(
                X_train,
                y_train,
                epochs=epochs_value,
                verbose=verbose_value,
                validation_split=validation_split_value,
                callbacks=[callback],
                use_multiprocessing=True,
            )
            model.summary()
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return model, history

    @staticmethod
    def ml_model_predict(model_name, X_feature_test):
        y_label_predicted = None
        try:
            y_label_predicted = model_name.predict(X_feature_test)
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return y_label_predicted

    @staticmethod
    def calculate_classification_metrics(y_original, y_predicted):
        accuracy_score_value = precision_value = recall_value = f1_score_value = (
            confusion_matrix_value
        ) = classification_report_value = None
        try:
            accuracy_score_value = float(
                "{0:0.3f}".format(accuracy_score(y_original, y_predicted) * 100)
            )
            precision_value = float(
                "{0:0.3f}".format(
                    precision_score(y_original, y_predicted, average="weighted") * 100
                )
            )
            recall_value = float(
                "{0:0.3f}".format(
                    recall_score(y_original, y_predicted, average="weighted") * 100
                )
            )
            f1_score_value = float(
                "{0:0.3f}".format(
                    f1_score(y_original, y_predicted, average="weighted") * 100
                )
            )
            confusion_matrix_value = confusion_matrix(y_original, y_predicted)
            classification_report_value = classification_report(y_original, y_predicted)
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return (
            accuracy_score_value,
            precision_value,
            recall_value,
            f1_score_value,
            confusion_matrix_value,
            classification_report_value,
        )

    @staticmethod
    def imbalanced_classes_plot(
        ds_y, is_class_number, class_column_name, x_label, y_label, title, font_size=8
    ):
        try:
            plt.figure(figsize=(5, 5))
            sns.set(style="darkgrid")
            if is_class_number == True:
                ax = sns.countplot(x=ds_y)
            else:
                ax = sns.countplot(x=class_column_name, data=ds_y)
            ax.set_xlabel(x_label, fontsize=font_size)
            ax.set_ylabel(y_label, fontsize=font_size)
            ax.tick_params(labelsize=font_size)
            for p in ax.patches:
                height = p.get_height()
                ax.text(
                    p.get_x() + p.get_width() / 2.0,
                    height + 10,
                    "{:1.0f}".format(height),
                    ha="center",
                    fontsize=font_size,
                )
            plt.title(title, fontsize=font_size)
            plt.show()
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())

    @staticmethod
    def imbalanced_classes_rna_plot(
        ds_y, class_column_name, x_label, y_label, title, font_size=8
    ):
        try:
            plt.figure(figsize=(5, 5))
            sns.set(style="darkgrid")
            ax = sns.countplot(x=class_column_name, data=ds_y)
            ax.set_xlabel(x_label, fontsize=font_size)
            ax.set_ylabel(y_label, fontsize=font_size)
            ax.tick_params(labelsize=font_size)
            for p in ax.patches:
                height = p.get_height()
                ax.text(
                    p.get_x() + p.get_width() / 2.0,
                    height + 1,
                    "{:1.0f}".format(height),
                    ha="center",
                    fontsize=font_size,
                )
            plt.title(title, fontsize=font_size)
            plt.show()
        except:
            exception_message = sys.exc_info()[0]
            print("An error occurred. {}".format(exception_message))

    @staticmethod
    def dna_sequence_is_equal_length(dna_sequence_series):
        try:
            dna_same_length = all(
                len(sequence) == len(dna_sequence_series[0])
                for sequence in dna_sequence_series
            )
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return dna_same_length

    @staticmethod
    def cnn_X_onehot_encoder(X):
        try:
            X_list_encoded = []
            X = X.tolist()
            for x in X:
                label_encoder = LabelEncoder()
                onehot_encoder = OneHotEncoder(sparse=False)
                x_onehot_encoded = PyDNA.dna_X_onehot_encoder(
                    label_encoder, onehot_encoder, x
                )
                X_list_encoded.append(x_onehot_encoded)
            X_onehot_encoded = np.stack(X_list_encoded)
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return X_onehot_encoded

    @staticmethod
    def dna_X_onehot_encoder(label_encoder, onehot_encoder, dna_sequence):
        try:
            dna_sequence = list(dna_sequence)
            dna_sequence_labelencoder = label_encoder.fit_transform(dna_sequence)
            dna_sequence_reshape = np.array(dna_sequence_labelencoder).reshape(-1, 1)
            dna_sequence_onehotencoder = onehot_encoder.fit_transform(
                dna_sequence_reshape
            )
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return dna_sequence_onehotencoder

    @staticmethod
    def dna_y_onehot_encoder(onehot_encoder, dna_label):
        try:
            dna_label_reshape = np.array(dna_label).reshape(-1, 1)
            dna_label_onehotencoder = onehot_encoder.fit_transform(dna_label_reshape)
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return dna_label_onehotencoder

    @staticmethod
    def cnn_y_onehot_encoder(y):
        try:
            y = y.tolist()
            onehot_encoder = OneHotEncoder(sparse=False)
            y_onehot_encoder = PyDNA.dna_y_onehot_encoder(onehot_encoder, y)
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return y_onehot_encoder

    @staticmethod
    def get_max_nparray(nparray):
        try:
            max_value = np.argmax(nparray, axis=1)
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return max_value

    @staticmethod
    def cnn_model_loss_plot(
        cnn_history, font_size, title_label, x_label, y_label, legend_label
    ):
        try:
            plt.figure(figsize=[5, 5])
            plt.style.use("ggplot")
            plt.tick_params(labelsize=font_size)
            plt.plot(cnn_history.history["loss"])
            plt.plot(cnn_history.history["val_loss"])
            plt.title(title_label, fontsize=font_size)
            plt.xlabel(x_label, fontsize=font_size)
            plt.ylabel(y_label, fontsize=font_size)
            plt.legend(legend_label, fontsize=font_size)
            plt.show()
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())

    @staticmethod
    def cnn_model_accuracy_plot(
        cnn_history, font_size, title_label, x_label, y_label, legend_label
    ):
        try:
            plt.figure(figsize=[5, 5])
            plt.style.use("ggplot")
            plt.tick_params(labelsize=font_size)
            # plt.plot(cnn_history.history["binary_accuracy"])
            # plt.plot(cnn_history.history["val_binary_accuracy"])
            plt.plot(cnn_history.history["accuracy"])
            plt.plot(cnn_history.history["val_accuracy"])
            plt.title(title_label, fontsize=font_size)
            plt.xlabel(x_label, fontsize=font_size)
            plt.ylabel(y_label, fontsize=font_size)
            plt.legend(legend_label, fontsize=font_size)
            plt.show()
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())

    @staticmethod
    def lstm_model_accuracy_plot(
        cnn_history, font_size, title_label, x_label, y_label, legend_label
    ):
        try:
            plt.figure(figsize=[5, 5])
            plt.style.use("ggplot")
            plt.tick_params(labelsize=font_size)
            plt.plot(cnn_history.history["accuracy"])
            plt.plot(cnn_history.history["val_accuracy"])
            plt.title(title_label, fontsize=font_size)
            plt.xlabel(x_label, fontsize=font_size)
            plt.ylabel(y_label, fontsize=font_size)
            plt.legend(legend_label, fontsize=font_size)
            plt.show()
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())

    @staticmethod
    def cnn_model_save_h5(cnn_model, cnn_model_path, cnn_model_name):
        try:
            model_path_name = os.path.join(cnn_model_path, cnn_model_name)
            cnn_model.save(model_path_name)
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())

    @staticmethod
    def cnn_model_load_h5(cnn_model_path, cnn_model_name):
        try:
            model_path_name = os.path.join(cnn_model_path, cnn_model_name)
            cnn_model = load_model(model_path_name)
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return cnn_model

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def read_app_config_file(section_name, option_name):
        """
        read from application configuration file
        :param section_name: section header name
        :param option_name: option value
        """
        option_value = None
        try:
            config_parser = configparser.ConfigParser()
            config_parser.read(PyDNA._app_config_file)
            option_value = config_parser.get(section_name, option_name)
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        return option_value

    @staticmethod
    def write_app_config_file(section_name, option_name, option_value):
        """
         write to application configuration file
        :param section_name: section header name
        :param option_name: option name
        :param option_value: option value
        """
        try:
            config_parser = configparser.ConfigParser()
            config_parser.read(PyDNA._app_config_file)
            if not config_parser.has_section(section_name):
                config_parser.add_section(section_name)
            config_parser.set(section_name, option_name, option_value)
            with open(PyDNA._app_config_file, "w") as open_config_file:
                config_parser.write(open_config_file)
        except:
            print(PyDNA.get_exception_info())
            if PyDNA._app_is_log:
                PyDNA.write_log_file("error", PyDNA.get_exception_info())
        finally:
            if open_config_file is not None:
                open_config_file.close()

    @staticmethod
    def write_log_file(event_level, message):
        """
         write messages to a log file
        :param event_level: even level name (CRITICAL, DEBUG, ERROR, INFO and WARNING)
        :param message: message to be written
        :return: None
        """
        try:
            logging_format_message = PyDNA.read_app_config_file(
                "logging_format", "logging_format_message"
            )
            logging_format_datetime = PyDNA.read_app_config_file(
                "logging_format", "logging_format_datetime"
            )
            if event_level == "CRITICAL".lower():
                logging.basicConfig(
                    format=logging_format_message,
                    datefmt=logging_format_datetime,
                    filename=PyDNA._app_log_file,
                    level=logging.CRITICAL,
                )
                logging.critical(message)
            elif event_level == "ERROR".lower():
                logging.basicConfig(
                    format=logging_format_message,
                    datefmt=logging_format_datetime,
                    filename=PyDNA._app_log_file,
                    level=logging.ERROR,
                )
                logging.error(message)
            elif event_level == "WARNING".lower():
                logging.basicConfig(
                    format=logging_format_message,
                    datefmt=logging_format_datetime,
                    filename=PyDNA._app_log_file,
                    level=logging.WARNING,
                )
                logging.warning(message)
            elif event_level == "INFO".lower():
                logging.basicConfig(
                    format=logging_format_message,
                    datefmt=logging_format_datetime,
                    filename=PyDNA._app_log_file,
                    level=logging.INFO,
                )
                logging.info(message)
            elif event_level == "DEBUG".lower():
                logging.basicConfig(
                    format=logging_format_message,
                    datefmt=logging_format_datetime,
                    filename=PyDNA._app_log_file,
                    level=logging.DEBUG,
                )
                logging.debug(message)
        except:
            print(PyDNA.get_exception_info())

    @staticmethod
    def get_exception_info(is_time_stamp=None):
        """
        get full stack exception error: file name, procedure name, error message, error type, error line number, error line code
        param is_time_stamp: include the time stamp or not
        return: full stack exception information
        """
        exception_info = None
        try:
            exception_type, exception_value, exception_traceback = sys.exc_info()
            file_name, line_number, procedure_name, line_code = traceback.extract_tb(
                exception_traceback
            )[-1]
            if is_time_stamp is not None:
                exception_info = "".join(
                    "[Time Stamp]: "
                    + str(time.strftime("%d-%m-%Y %I:%M:%S %p"))
                    + " "
                    + "[File Name]: "
                    + str(file_name)
                    + " "
                    + "[Procedure Name]: "
                    + str(procedure_name)
                    + " "
                    + "[Error Message]: "
                    + str(exception_value)
                    + " "
                    + "[Error Type]: "
                    + str(exception_type)
                    + " "
                    + "[Line Number]: "
                    + str(line_number)
                    + " "
                    + "[Line Code]: "
                    + str(line_code)
                )
            else:
                exception_info = "".join(
                    "[File Name]: "
                    + str(file_name)
                    + " "
                    + "[Procedure Name]: "
                    + str(procedure_name)
                    + " "
                    + "[Error Message]: "
                    + str(exception_value)
                    + " "
                    + "[Error Type]: "
                    + str(exception_type)
                    + " "
                    + "[Line Number]: "
                    + str(line_number)
                    + " "
                    + "[Line Code]: "
                    + str(line_code)
                )
        except:
            print("An error occurred in {}.".format("get_exception_info() function."))
        return exception_info

    @staticmethod
    def pickle_serialize_object(file_path_name, data_object):
        """serialize an object
        Args:
            file_path_name ([type]): [description]
            data_object ([type]): [description]
        """
        try:
            with open(file_path_name, "wb") as data_outfile:
                pkl.dump(data_object, data_outfile)
        except:
            pass

    @staticmethod
    def pickle_deserialize_object(file_path_name):
        """deserialize an object
        Args:
            file_path_name ([type]): [description]
        Returns:
            [type]: [description]
        """
        data_object = None
        try:
            with open(file_path_name, "rb") as data_infile:
                data_object = pkl.load(data_infile)
        except:
            pass
        return data_object

    @staticmethod
    def is_string_empty_none(value_string):
        """_summary_

        Args:
            value_string (_type_): _description_

        Returns:
            _type_: _description_
        """
        try:
            if not (value_string and not value_string.isspace()) or (
                value_string is None
            ):
                is_empty_none = True
            else:
                is_empty_none = False
        except:
            pass
        return is_empty_none


class ValAccuracyCallback(tf.keras.callbacks.Callback):
    """_summary_

    Args:
        tf (_type_): _description_
    """

    def __init__(self, val_accuracy_threshold):
        self.val_accuracy_threshold = val_accuracy_threshold

    def on_epoch_end(self, epoch, logs=None):
        """_summary_

        Args:
            epoch (_type_): _description_
            logs (_type_, optional): _description_. Defaults to None.
        """
        val_acc = logs["val_accuracy"]
        if val_acc >= self.val_accuracy_threshold:
            self.model.stop_training = True

    class __metaclass__(type):
        @property
        def app_config_file(cls):
            return cls._app_config_file

        @app_config_file.setter
        def app_config_file(cls, value):
            cls._app_config_file = value

        @property
        def app_log_file(cls):
            return cls._app_log_file

        @app_log_file.setter
        def app_log_file(cls, value):
            cls._app_log_file = value
