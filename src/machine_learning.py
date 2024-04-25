
import pandas as pd
from scipy.io import arff
from src.constantes import ConstantesArff as coarff
from lazypredict.Supervised import LazyClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2


class ArffToDf:
    def __init__(self):
        self.arff_file_train = arff.loadarff(coarff.PATH_ARFF_TRAIN.value)
        self.arff_file_test = arff.loadarff(coarff.PATH_ARFF_TEST.value)

    def dataframe(self):
        df_train = pd.DataFrame(self.arff_file_train[0])
        df_test = pd.DataFrame(self.arff_file_test[0])
        self.colonnes = df_train.columns.tolist()
        return df_train, df_test

    def x_y_split(self, dataframe):
        X = dataframe.iloc[:, :-1]
        y = dataframe.iloc[:, -1]
        return X, y

    def encode_X(self, dataframe):
        df_object = dataframe.select_dtypes(include='object')
        df_encode = pd.get_dummies(dataframe, columns=df_object.columns)
        df_encode.columns = df_encode.columns.str.replace("b'", "")
        df_encode.columns = df_encode.columns.str.replace("'", "")
        return df_encode

    def encode_y(self, y):
        le = LabelEncoder()
        y = le.fit_transform(y)
        return y

    def egalisage_colonne(self, X_train, X_test):
        missing_columns = set(X_train.columns) - set(X_test.columns)
        for column in missing_columns:
            X_test[column] = 0
        X_test = X_test[X_train.columns]  # permet de remettre les colonnes dans le bon ordre
        return X_test

    def conversion_numpy(self, X_train, X_test):
        X_train = X_train.to_numpy()
        X_test = X_test.to_numpy()
        return X_train, X_test

    def process(self):
        df_train, df_test = self.dataframe()
        X_train, y_train = self.x_y_split(df_train)
        X_test, y_test = self.x_y_split(df_test)
        X_train = self.encode_X(X_train)
        X_test = self.encode_X(X_test)
        y_train = self.encode_y(y_train)
        y_test = self.encode_y(y_test)
        X_test = self.egalisage_colonne(X_train, X_test)
        return X_train, y_train, X_test, y_test, self.colonnes


X_train, y_train, X_test, y_test, colonnes = ArffToDf().process()
clf = LazyClassifier(verbose=True, ignore_warnings=False, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)


class TraitementDonnees:
    def __init__(self, X):
        self.X = X

    def pca_election(self):
        pca = PCA()

    def chi_selection(self, X, y):
        chi2_selector = SelectKBest(chi2, k=10)
        X_kbest = chi2_selector.fit_transform(X, y)


