
import pandas as pd
from scipy.io import arff
from constantes import ConstantesArff as coarff



class ArffToDf:
    def __init__(self):
        self.arff_file_train = arff.loadarff(coarff.PATH_ARFF_TRAIN.value)
        self.arff_file_test = arff.loadarff(coarff.PATH_ARFF_TEST.value)

    def dataframe(self):
        df_train = pd.DataFrame(self.arff_file_train[0])
        df_test = pd.DataFrame(self.arff_file_test[0])
        return df_test


df_test = ArffToDf().dataframe()
print(df_test)
