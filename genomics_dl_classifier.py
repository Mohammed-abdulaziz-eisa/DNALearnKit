import sys
import time
import os

os.system("cls")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

tf.random.set_seed(10)

import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from torch import lstm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from pydna import PyDNA
from pydna import *

import warnings

warnings.filterwarnings("ignore")


def get_program_running(start_time):
    end_time = time.process_time()
    diff_time = end_time - start_time
    result = time.strftime("%H:%M:%S", time.gmtime(diff_time))
    print("program runtime: {}".format(result))


# deep_learning_genomics_primer_tutorial_colab_cnn_lstm_ernest_refactor_final_paper.py
def main():
    print(
        "deep_learning_genomics_primer_tutorial_colab_cnn_lstm_ernest_refactor_final_paper"
    )
    # csv file path and name
    csv_path_file = r"G:\Visual WWW\Python\1000_python_workspace\genetic_diagnostics_deep_learning\oxnrous_sent_files\dna_sequence_protein.csv"
    df_genomics = PyDNA.pandas_read_data("CSV", csv_path_file, None)
    df_genomics.info()

    # remove rows and columns with missing values.
    df_genomics.dropna(how="all", inplace=True)
    # exit()

    # select X features
    X = PyDNA.select_df_column(df_genomics, "dna_sequence")
    # print(X)

    # check if dna sequences have the same legnth - part of dna sequence preprocessing!
    dna_is_same_length = PyDNA.dna_sequence_is_equal_length(X)
    if dna_is_same_length == False:
        print("program exit:")
        print(
            "dna sequences have the different legnth. deep learning CNN or LSTM can't be applied."
        )
        exit()
    else:
        print(
            "dna sequences have the same legnth. deep learning CNN or LSTM can be applied."
        )

    # X features one-hot encoder
    X = PyDNA.cnn_X_onehot_encoder(X)
    # print(X)

    # select y label
    y = PyDNA.select_df_column(df_genomics, "dna_label")
    # print(y)

    # show y label imbalanced classes plot
    # PyDNA.imbalanced_classes_plot(y, True, "dna_label", "DNA bind to protein class", "Count", "DNA Sequence Protein Classes")

    # y label one-hot encoder
    y = PyDNA.cnn_y_onehot_encoder(y)
    # print(y)
    # exit()

    # data split in train, valid and test (80%/10%/10%)
    X_train, y_train, X_valid, y_valid, X_test, y_test = (
        PyDNA.train_validation_test_split(X, y, test_size=0.2, valid_size=0.5)
    )

    # FOR CNN ----------------------------------------------------------------------------------------------------
    # print("CNN")
    # epochs_number = 50
    # data_split = 0.2
    # val_accuracy_threshold = 0.99
    # cnn_model, cnn_history = PyDNA.create_cnn_model(y_train, X_train, epochs_number, data_split, val_accuracy_threshold, verbose_value=1)
    # # plot cnn model loss
    # font_size = 8
    # PyDNA.cnn_model_loss_plot(cnn_history, font_size, "CNN Model Loss", "Epoch", "Loss", ["Train", "Validation"])
    # # plot cnn model accuracy
    # PyDNA.cnn_model_accuracy_plot(cnn_history, font_size, "CNN Model Accuracy", "Epoch", "Accuracy", ["Train", "Validation"])
    # ------------------------------------------------------------------------------------------------------------------

    # FOR LSTM ------------------------------------------------------------------------------------------------
    print("LSTM")
    epochs_number = 50
    data_split = 0.2
    val_accuracy_threshold = 0.99
    cnn_model, cnn_history = PyDNA.create_lstm_model(
        y_train,
        X_train,
        epochs_number,
        data_split,
        val_accuracy_threshold,
        verbose_value=1,
    )
    # plot cnn model loss
    font_size = 8
    PyDNA.cnn_model_loss_plot(
        cnn_history,
        font_size,
        "LSTM Model Loss",
        "Epoch",
        "Loss",
        ["Train", "Validation"],
    )
    # plot cnn model accuracy
    PyDNA.lstm_model_accuracy_plot(
        cnn_history,
        font_size,
        "LSTM Model Accuracy",
        "Epoch",
        "Accuracy",
        ["Train", "Validation"],
    )
    # --------------------------------------------------------------------------------------------------------------

    print()
    print("model validation metrics")
    # get y_predicted valid
    y_predicted = cnn_model.predict(X_valid)

    # get max indices of the maximum values along an axis
    y_val_max = PyDNA.get_max_nparray(y_valid)
    y_predicted_max = PyDNA.get_max_nparray(y_predicted)

    # calculate valid classification metrics
    (
        accuracy_score_value,
        precision_value,
        recall_value,
        f1_score_value,
        confusion_matrix_value,
        classification_report_value,
    ) = PyDNA.calculate_classification_metrics(y_val_max, y_predicted_max)
    print("valid accuracy score:\n{}\n".format(accuracy_score_value))
    print("valid precision:\n{}\n".format(precision_value))
    print("valid recall:\n{}\n".format(recall_value))
    print("valid f1 score:\n{}\n".format(f1_score_value))
    print("valid confusion matrix:\n{}\n".format(confusion_matrix_value))
    print("valid classification report:\n{}\n".format(classification_report_value))

    print("model test")
    # get y_predicted test
    y_predicted = cnn_model.predict(X_test)
    # print(X_test.shape)

    # get max indices of the maximum values along an axis
    y_test_max = PyDNA.get_max_nparray(y_test)
    y_predicted_max = PyDNA.get_max_nparray(y_predicted)

    # calculate test classification metrics
    (
        accuracy_score_value,
        precision_value,
        recall_value,
        f1_score_value,
        confusion_matrix_value,
        classification_report_value,
    ) = PyDNA.calculate_classification_metrics(y_test_max, y_predicted_max)
    print("test accuracy score:\n{}\n".format(accuracy_score_value))
    print("test precision:\n{}\n".format(precision_value))
    print("test recall:\n{}\n".format(recall_value))
    print("test f1 score:\n{}\n".format(f1_score_value))
    print("test confusion matrix:\n{}\n".format(confusion_matrix_value))
    print("test classification report:\n{}\n".format(classification_report_value))

    # SAVE CNN MODEL WITH H5 FORMAT ------------------------------------------------------------------------------
    # cnn_model_path = r"G:\Visual WWW\Python\1000_python_workspace\genetic_diagnostics_deep_learning\h5"
    # cnn_model_name = "cnn_genomic_model.h5"
    # PyDNA.cnn_model_save_h5(cnn_model, cnn_model_path, cnn_model_name)
    # # load cnn model h5 for future use
    # cnn_model = PyDNA.cnn_model_load_h5(cnn_model_path, cnn_model_name)
    # -------------------------------------------------------------------------------------------------------------------------------------

    # SAVE LSTM MODEL WITH H5 FORMAT------------------------------------------------------------------------------
    lstm_model_path = r"G:\Visual WWW\Python\1000_python_workspace\genetic_diagnostics_deep_learning\h5"
    lstm_model_name = "lstm_genomic_model.h5"
    PyDNA.cnn_model_save_h5(cnn_model, lstm_model_path, lstm_model_name)
    # load cnn model h5 for future use
    cnn_model = PyDNA.cnn_model_load_h5(lstm_model_path, lstm_model_name)

    # ------------------------------------------------------------------------------------------------------------
    # real-production model one test
    # print()
    # print("real-production model one test")
    # csv_path_file = os.path.join(csv_path_folder, "dna_sequence_protein_one_test.csv")

    # df_genomics = PyDNA.pandas_read_data("CSV", csv_path_file, None)
    # df_genomics.info()

    # X_real = PyDNA.select_df_column(df_genomics, "dna_sequence")
    # # print(X_real)

    # X_real = PyDNA.cnn_X_onehot_encoder(X_real)
    # # print(X)
    # # print(X.shape)

    # y_predicted = cnn_model.predict(X_real)
    # # print(X_real.shape)

    # # get max indices of the maximum values along an axis
    # y_test_max = PyDNA.get_max_nparray(y_test)
    # y_predicted_max = PyDNA.get_max_nparray(y_predicted)
    # print(y_predicted_max)


if __name__ == "__main__":
    start_time = time.process_time()
    main()
    get_program_running(start_time)

# dna sequences tests
# ACTCGCTGTCCACGTCTATTCCTAGGGGTTTTATTTCGCAAGGTGATACT,0
# TGCAAAGGGGCGACCGAACTCCCTTTACCGCGGAGTTATTCATAATTGAA,1

# FOR CNN KERAS-TF
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv1d (Conv1D)              (None, 39, 32)            1568
# _________________________________________________________________
# max_pooling1d (MaxPooling1D) (None, 9, 32)             0
# _________________________________________________________________
# flatten (Flatten)            (None, 288)               0
# _________________________________________________________________
# dense (Dense)                (None, 16)                4624
# _________________________________________________________________
# dense_1 (Dense)              (None, 2)                 34
# =================================================================
# Total params: 6,226
# Trainable params: 6,226
# Non-trainable params: 0
# _________________________________________________________________
# classification accuracy score:
# 98.0

# classification confusion matrix:
# [[195   4]
#  [  4 197]]

# classification report:
#               precision    recall  f1-score   support

#            0       0.98      0.98      0.98       199
#            1       0.98      0.98      0.98       201

#     accuracy                           0.98       400
#    macro avg       0.98      0.98      0.98       400
# weighted avg       0.98      0.98      0.98       400
