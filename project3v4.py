# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import datetime
import argparse
from contextlib import contextmanager


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingRegressor

'''
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.models import Sequential
'''

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, SelectPercentile
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.utils import check_random_state


from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import AtomPairs
from rdkit.Chem.AtomPairs import Torsions
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.AllChem import GetHashedAtomPairFingerprintAsBitVect
from rdkit.Chem import MACCSkeys

import random
from math import floor

import pandas as pd


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

    #nBits = 167
    nBits = 512
    X = np.zeros((n_chem, 512))
    X2 = np.zeros((n_chem, 167))
    X3 = np.zeros((n_chem, 2048))
    X4 = np.zeros((n_chem, 2048))
    for i in range(n_chem):
        m = Chem.MolFromSmiles(chemical_compounds[i])
        #X[i,:] = calcfp("maccs", m)
        X[i,:] = AllChem.GetMorganFingerprintAsBitVect(m,3,nBits=512,useFeatures=True)
        X2[i,:] = Chem.MACCSkeys.GenMACCSKeys(m)
        X3[i,:] = Chem.RDKFingerprint(m)
        X4[i,:] = Chem.LayeredFingerprint(m)
        #print(AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=1024))
    Xret = np.concatenate((X,X2,X3,X4),axis=1)
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Make a toy submission")
    parser.add_argument("--ls", default="data/training_set.csv",
                        help="Path to the learning set as CSV file")
    parser.add_argument("--ts", default="data/test_set.csv",
                        help="Path to the test set as CSV file")
    parser.add_argument("--dt", action="store_true", default=False,
                        help="Use a decision tree classifier (by default, "
                             "make a random prediction)")
    parser.add_argument("--oth", action="store_true", default=False)
    parser.add_argument("--nn", action="store_true", default=False)

    args = parser.parse_args()

    # Load training data
    LS = load_from_csv(args.ls)
    # Load test data
    TS = load_from_csv(args.ts)

    if args.dt:
        # -------------------------- Decision Tree --------------------------- #

        # LEARNING
        # Create fingerprint features and output
        with measure_time("Creating fingerprint"):
            X_LS = create_fingerprints(LS["SMILES"].values)
        y_LS = LS["ACTIVE"].values

        # Build the model
        model = DecisionTreeClassifier()

        with measure_time('Training'):
            model.fit(X_LS, y_LS)

        # PREDICTION
        TS = load_from_csv(args.ts)
        X_TS = create_fingerprints(TS["SMILES"].values)

        # Predict
        y_pred = model.predict_proba(X_TS)[:,1]

        # Estimated AUC of the model
        auc_predicted = 0.50 # it seems a bit pessimistic, right?

        # Making the submission file
        fname = make_submission(y_pred, auc_predicted, 'toy_submission_DT')
        print('Submission file "{}" successfully written'.format(fname))

    elif args.oth:
        with measure_time("Creating fingerprint"):
            X_LS = create_fingerprints(LS["SMILES"].values)
        y_LS = LS["ACTIVE"].values

        model = RandomForestClassifier(n_estimators=100)

        model.fit(X_LS, y_LS)
        TS = load_from_csv(args.ts)
        X_TS = create_fingerprints(TS["SMILES"].values)
        y_pred = model.predict_proba(X_TS)[:,1]

        fname = make_submission(y_pred, 0.5, 'randomforest')
        print('Submission file "{}" successfully written'.format(fname))

        """model = Sequential()
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_LS, y_LS, epochs=120, batch_size=100)


        TS = load_from_csv(args.ts)
        X_TS = create_fingerprints(TS["SMILES"].values)

        y_pred = model.predict(X_TS)
        y_pred = [round(y[0]) for y in y_pred]

        fname = make_submission(y_pred, 0, 'keras')
        print('Submission file "{}" successfully written'.format(fname))"""

    elif args.nn:
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

    else:

        # ------------------------ Random Prediction ------------------------- #
        # Predict
        random_state = 0
        random_state = check_random_state(random_state)
        y_pred = random_state.rand(TS.shape[0])

        # Estimated AUC of the model
        auc_predicted = 0.50 # expected value for random guessing

        # Making the submission file
        fname = make_submission(y_pred, auc_predicted, 'toy_submission_random')
        print('Submission file "{}" successfully written'.format(fname))
