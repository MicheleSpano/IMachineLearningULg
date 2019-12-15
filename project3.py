# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import datetime
import argparse
from contextlib import contextmanager
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.utils import check_random_state
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
import random



@contextmanager
def measure_time(label):
    """
    Context manager to measure time of computation.
    >>> with measure_time('Heavy computation'):
    >>>     do_heavy_computation()
    'Duration of [Heavy computation]: 0:04:07.765971'

    Parameters
    ----------
    label: str
        The label by which the computation will be referred
    """
    print("{}...".format(label))
    start = time.time()
    yield
    end = time.time()
    print('Duration of [{}]: {}'.format(label,
                                        datetime.timedelta(seconds=end-start)))


def load_from_csv(path, delimiter=','):
    """
    Load csv file and return a NumPy array of its data

    Parameters
    ----------
    path: str
        The path to the csv file to load
    delimiter: str (default: ',')
        The csv field delimiter

    Return
    ------
    D: array
        The NumPy array of the data contained in the file
    """
    if not os.path.exists(path):
        raise FileNotFoundError("File '{}' does not exists.".format(path))
    return pd.read_csv(path, delimiter=delimiter)




def create_fingerprints(chemical_compounds, fptype="rdkit"):
    """
    Create a learning matrix `X` with (Morgan) fingerprints
    from the `chemical_compounds` molecular structures.

    Parameters
    ----------
    chemical_compounds: array [n_chem, 1] or list [n_chem,]
        chemical_compounds[i] is a string describing the ith chemical
        compound.

    Return
    ------
    X: array [n_chem, 124]
        Generated (Morgan) fingerprints for each chemical compound, which
        represent presence or absence of substructures.
    """
    n_chem = chemical_compounds.shape[0]

    X = np.zeros((n_chem, 512))
    X2 = np.zeros((n_chem, 167))
    X3 = np.zeros((n_chem, 2048))
    X4 = np.zeros((n_chem, 2048))
    for i in range(n_chem):
        m = Chem.MolFromSmiles(chemical_compounds[i])
        X[i,:] = AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=512, useFeatures=True)
        X2[i,:] = Chem.MACCSkeys.GenMACCSKeys(m)
        X3[i,:] = Chem.RDKFingerprint(m)
        X4[i,:] = Chem.LayeredFingerprint(m)
    Xret = np.concatenate((X,X2,X3,X4), axis=1)
    return Xret


def make_submission(y_predicted, auc_predicted, file_name="submission", date=True, indexes=None):
    """
    Write a submission file for the Kaggle platform

    Parameters
    ----------
    y_predicted: array [n_predictions, 1]
        if `y_predict[i]` is the prediction
        for chemical compound `i` (or indexes[i] if given).
    auc_predicted: float [1]
        The estimated ROCAUC of y_predicted.
    file_name: str or None (default: 'submission')
        The path to the submission file to create (or override). If none is
        provided, a default one will be used. Also note that the file extension
        (.txt) will be appended to the file.
    date: boolean (default: True)
        Whether to append the date in the file name

    Return
    ------
    file_name: path
        The final path to the submission file
    """

    # Naming the file
    if date:
        file_name = '{}_{}'.format(file_name, time.strftime('%d-%m-%Y_%Hh%M'))

    file_name = '{}.txt'.format(file_name)

    # Creating default indexes if not given
    if indexes is None:
        indexes = np.arange(len(y_predicted))+1

    # Writing into the file
    with open(file_name, 'w') as handle:
        handle.write('"Chem_ID","Prediction"\n')
        handle.write('Chem_{:d},{}\n'.format(0,auc_predicted))

        for n,idx in enumerate(indexes):

            if np.isnan(y_predicted[n]):
                raise ValueError('The prediction cannot be NaN')
            line = 'Chem_{:d},{}\n'.format(idx, y_predicted[n])
            handle.write(line)
    return file_name

def plot_roc_auc(labels, predictions):
    fpr, tpr, _ = roc_curve(labels, predictions)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % best_model["auc"])
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
    print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
    print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
    print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
    print('Total Fraudulent Transactions: ', np.sum(cm[1]))
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Make a toy submission")
    parser.add_argument("--ls", default="data/training_set.csv",
                        help="Path to the learning set as CSV file")
    parser.add_argument("--ts", default="data/test_set.csv",
                        help="Path to the test set as CSV file")

    args = parser.parse_args()

    # Load training data
    LS = load_from_csv(args.ls)
    # Load test data
    TS = load_from_csv(args.ts)

    # Fingerprint creation
    with measure_time("Creating learning sample fingerprint"):
        X_WholeSet = create_fingerprints(LS["SMILES"].values)
    y_WholeSet = LS["ACTIVE"].values

    TS = load_from_csv(args.ts)
    with measure_time("Creating test sample fingerprint"):
        X_TS = create_fingerprints(TS["SMILES"].values)

    # Feature selection
    _k = 1000
    with measure_time("Selecting the "+str(_k)+" best features"):
        featureSelector = SelectKBest(chi2, k=_k)
        X_WholeSet = featureSelector.fit_transform(X_WholeSet, y_WholeSet)
        X_TS = featureSelector.transform(X_TS)

    # Splitting
    with measure_time("Splitting and shuffling the datas"):
        X_LS, X_VS, y_LS, y_VS = train_test_split(X_WholeSet, y_WholeSet, test_size=0.33, random_state=42)


    WS = list(zip(X_WholeSet, y_WholeSet))
    LS = list(zip(X_LS, y_LS))
    VS = list(zip(X_VS, y_VS))

    # Changing 0 to 1 ratio in learning samples
    ratio = 9
    LS0 = []
    LS1 = []
    for (x, y) in LS:
        if not y:
            LS0 += [(x, y)]
        else:
            LS1 += [(x, y)]
    LS0 = LS0[0:round(len(LS1)*ratio)]
    LS = LS0 + LS1

    random.shuffle(WS)
    random.shuffle(LS)
    random.shuffle(VS)

    X_WS, y_WS = zip(*WS)
    X_LS, y_LS = zip(*LS)
    X_VS, y_VS = zip(*VS)

    # Model creation and assessment
    models=[]
    with measure_time("Creating and assessing the models"):
        for param1 in [8000]:
            for param2 in ['balanced']:
                for param3 in [None]:
                    for param4 in ["not defined"]:
                        model = RandomForestClassifier(n_estimators=param1, class_weight=param2, max_depth=param3)
                        model.fit(X_LS, y_LS)

                        y_predict = model.predict_proba(X_VS)[:,1]
                        auc = roc_auc_score(y_VS, y_predict)

                        models += [{"model": model,
                                    "param1": param1,
                                    "param2": param2,
                                    "param3": param3,
                                    "param4": param4,
                                    "auc": auc
                                    }]

    # Finding the best model
    best_model = models[0]
    for model in models:
        if(model["auc"] >= best_model["auc"]):
            best_model = model


    print("------BEST MODEL-------")
    print("param1="+str(best_model["param1"]))
    print("param2="+str(best_model["param2"]))
    print("param3="+str(best_model["param3"]))
    print("param4="+str(best_model["param4"]))
    print("auc="+str(best_model["auc"]))
    print("-----------------------")


    # Plotting charts
    plot_cm(y_VS, y_predict)
    plot_roc_auc(y_VS, y_predict)


    # Best model fitting
    with measure_time("Fitting the best model"):
        best_model["model"].fit(X_WS, y_WS)


    # Changing 0 to 1 ratio in final sample
    WS = list(zip(X_WS, y_WS))
    WS0 = []
    WS1 = []
    for (x, y) in WS:
        if not y:
            WS0 += [(x, y)]
        else:
            WS1 += [(x, y)]
    WS0 = WS0[0:round(len(WS1)*ratio)]
    WS = WS0 + WS1
    random.shuffle(WS)

    # Predicting and submitting
    y_pred = best_model["model"].predict_proba(X_TS)[:,1]

    fname = make_submission(y_pred, best_model["auc"]-0.06, 'random_forest')
    print('Submission file "{}" successfully written'.format(fname))
    print("Done")