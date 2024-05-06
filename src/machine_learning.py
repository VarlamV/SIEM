
import pandas as pd
from scipy.io import arff
from src.constantes import ConstantesArff as coarff
# from lazypredict.Supervised import LazyClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, make_scorer

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


def chi_selection(X_train, X_test, y_train, n):
    chi2_selector = SelectKBest(chi2, k=n)
    X_train_transfo = chi2_selector.fit_transform(X_train, y_train)
    X_test_transfo = chi2_selector.transform(X_test)
    mask = chi2_selector.get_support()
    colonnes_prises = X_train.columns[mask]
    return X_train_transfo, X_test_transfo, colonnes_prises


X_train, y_train, X_test, y_test, colonnes = ArffToDf().process()

#clf = LazyClassifier(verbose=True, ignore_warnings=False, custom_metric=None)
#models, predictions = clf.fit(X_train, X_test, y_train, y_test)

liste_X_train = []
liste_X_test = []
liste_chi = []
liste_colonnes = []
for i in range(5, 80):
    X_train_new, X_test_new, colonnes_best = chi_selection(X_train, X_test, y_train, i)
    liste_X_train.append(X_train_new)
    liste_X_test.append(X_test_new)
    liste_chi.append(i)
    liste_colonnes.append(colonnes_best)

liste_accury = []

for xtrain, xtest in zip(liste_X_train, liste_X_test):
    clf = LGBMClassifier()
    clf.fit(xtrain, y_train)
    predictions = clf.predict(xtest)
    accuracy = accuracy_score(y_test, predictions)
    liste_accury.append(accuracy)

old_X_train = X_train
old_X_test = X_test

df_meilleur_chi = pd.DataFrame({'Nb de chi': liste_chi, 'Accuracy': liste_accury, 'Colonnes': liste_colonnes})
df_meilleur_chi.to_csv(coarff.PATH_CHI.value)

df_meilleur_chi = pd.read_csv(coarff.PATH_CHI.value)
index_best_du_best = df_meilleur_chi['Accuracy'].idxmax()
meilleur_chi = df_meilleur_chi.loc[index_best_du_best, 'Nb de chi']
meilleur_colonnes = df_meilleur_chi.loc[index_best_du_best, 'Colonnes']

X_train = liste_X_train[index_best_du_best]
X_test = liste_X_test[index_best_du_best]

param_grid = {
    'num_leaves': [],
    'max_depth': [],
    'learning_rate': [0.01, 0.1, 0.5]
}
for i in range(10, 32):
    param_grid['num_leaves'].append(i)
for g in range(5, 20):
    param_grid['max_depth'].append(g)


scoring = {'Accuracy': make_scorer(accuracy_score), 'F1': make_scorer(f1_score)}

clf = LGBMClassifier()
grid_search = GridSearchCV(clf, param_grid, scoring=scoring, refit='F1', cv=5, verbose=1)

grid_search.fit(X_train, y_train)

predictions = grid_search.best_estimator_.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

results = pd.DataFrame(grid_search.cv_results_)
results = results[['param_num_leaves', 'param_max_depth', 'param_learning_rate', 'mean_test_Accuracy', 'mean_test_F1']]
results.columns = ['Num Leaves', 'Max Depth', 'Learning Rate', 'Mean Accuracy', 'Mean F1 Score']

results = results.append({
    'Num Leaves': 'Best Model',
    'Max Depth': '',
    'Learning Rate': '',
    'Mean Accuracy': accuracy,
    'Mean F1 Score': f1
}, ignore_index=True)

clf = LGBMClassifier(num_leaves=31, max_depth=30, learning_rate=0.005)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
f_un = f1_score(y_test, predictions)

df = pd.read_csv('./data/chi.csv')
