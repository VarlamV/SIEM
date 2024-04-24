
import pandas as pd
from scipy.io import arff
from src.constantes import ConstantesArff as coarff


class ArffToDf:
    def __init__(self):
        self.arff_file_train = arff.loadarff(coarff.PATH_ARFF_TRAIN.value)
        self.arff_file_test = arff.loadarff(coarff.PATH_ARFF_TEST.value)

    def dataframe(self):
        df_train = pd.DataFrame(self.arff_file_train[0])
        df_test = pd.DataFrame(self.arff_file_test[0])
        return df_train, df_test

    def x_y_split(self, dataframe):
        X = dataframe.iloc[:, :-1]
        y = dataframe.iloc[:, -1]
        return X, y

    def encode_colonnes(self, dataframe):
        df_object = dataframe.select_dtypes(include='object')
        df_encode = pd.get_dummies(dataframe, columns=df_object.columns)
        return df_encode


    def process(self):
        df_train, df_test = self.dataframe()
        X_train, y_train = self.x_y_split(df_train)
        X_test, y_test = self.x_y_split(df_test)
        X_train = self.encode_colonnes(X_train)
        X_test = self.encode_colonnes(X_test)
        return X_train, y_train, X_test, y_test




X_train, y_train, X_test, y_test = ArffToDf().process()


class LeMonstre:
    def __init__(self):
        pass

    def pca_election(self):
        pass

    def chi_selection(self):
        pass


