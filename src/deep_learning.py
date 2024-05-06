
import pandas as pd
from scipy.io import arff
from constantes import ConstantesArff as coarff

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import EarlyStopping

import random


class ArffToDf:
    def __init__(self):
        self.arff_file_train = arff.loadarff(coarff.PATH_ARFF_TRAIN.value)
        self.arff_file_test = arff.loadarff(coarff.PATH_ARFF_TEST.value)

    def dataframe(self):
        df_train = pd.DataFrame(self.arff_file_train[0])
        df_test = pd.DataFrame(self.arff_file_test[0])
        return df_train,df_test


df_train,df_test = ArffToDf().dataframe()

for col in df_train.columns:
    if isinstance(df_train[col][1], bytes):
        df_train[col] = df_train[col].apply(lambda x: x.decode('utf-8'))

for col in df_test.columns:
    if isinstance(df_test[col][1], bytes):
        df_test[col] = df_test[col].apply(lambda x: x.decode('utf-8'))


df_train["land"] = df_train["land"].astype(float)
df_test["land"] = df_train["land"].astype(float)



df_train = df_train[df_train['service'] != "http_8001"]
df_train = df_train[df_train['service'] != "aol"]
df_train = df_train[df_train['service'] != "http_2784"]
df_train = df_train[df_train['service'] != "harvest"]
df_train = df_train[df_train['service'] != "red_i"]
df_train = df_train[df_train['service'] != "urh_i"]


X_train = df_train.iloc[:, :-1]
X_test = df_test.iloc[:, :-1]

derniere_colonne_train = df_train.iloc[:, -1]
derniere_colonne_test = df_test.iloc[:, -1]
y_train = pd.DataFrame(derniere_colonne_train, columns=['class'])
y_test = pd.DataFrame(derniere_colonne_test, columns=['class'])

print(X_train.shape)
print(X_test.shape)


X_train = pd.get_dummies(X_train, columns=["protocol_type","service","flag"])
X_test = pd.get_dummies(X_test, columns=["protocol_type","service","flag"])

missing_columns = set(X_train.columns) - set(X_test.columns)
print(missing_columns)

for column in missing_columns:
    X_test[column] = 0

X_test = X_test[X_train.columns]

exclus = [2,3] #exclu du drop
drop = [1,4]
print(X_train.columns[77],X_train.columns[29],X_train.columns[80],X_train.columns[67],X_train.columns[41],X_train.columns[48])
liste_drop = ["service_uucp_path", "service_pop_2", "service_daytime", "service_efs", "service_sql_net"]

# X_train = X_train.drop(columns=liste_drop)
# X_test = X_test.drop(columns=liste_drop)

print(X_train.columns)

# test_x = 1
# test_y = 3
#
#
# X_train = X_train.iloc[:, test_x:test_y]
# X_test = X_test.iloc[:, test_x:test_y]
#
# print(X_test)

# print(X_train.columns)

# nb1 = random.randint(0, 115)
# nb2 = random.randint(0, 115)
# nb3 = random.randint(0, 115)
# nb4 = random.randint(0, 115)
# nb5 = random.randint(0, 115)
#
# while nb1 in exclus:
#     nb1 = random.randint(0, 115)
#
# while nb2 in exclus:
#     nb2 = random.randint(0, 115)
#
# while nb3 in exclus:
#     nb3 = random.randint(0, 115)
#
# while nb4 in exclus:
#     nb4 = random.randint(0, 115)
#
# while nb5 in exclus:
#     nb5 = random.randint(0, 115)

# nb1 = 94
# nb2 = 73
# nb3 = 89
# nb4 = 109
# nb5 = 4

print("nbbbbbb")
# print(nb1,nb2,nb3,nb4,nb5)
#
# print(X_train.columns[nb1],X_train.columns[nb2],X_train.columns[nb3],X_train.columns[nb4],X_train.columns[nb5])
# print(X_train.columns[25],X_train.columns[41],X_train.columns[105],X_train.columns[3],X_train.columns[48])

# print(X_train.columns[91])
# X_train = X_train.drop(X_train.columns[nb1], axis=1)
# X_test = X_test.drop(X_test.columns[nb1], axis=1)
# print(X_train.columns[91])
# X_train = X_train.drop(X_train.columns[nb2], axis=1)
# X_test = X_test.drop(X_test.columns[nb2], axis=1)
#
# X_train = X_train.drop(X_train.columns[nb3], axis=1)
# X_test = X_test.drop(X_test.columns[nb3], axis=1)
#
# X_train = X_train.drop(X_train.columns[nb4], axis=1)
# X_test = X_test.drop(X_test.columns[nb4], axis=1)
#
# X_train = X_train.drop(X_train.columns[nb5], axis=1)
# X_test = X_test.drop(X_test.columns[nb5], axis=1)


print(X_train.shape)
print(X_test.shape)

label_encoder = LabelEncoder()

# Appliquer l'encodage des étiquettes sur la colonne catégorielle
y_train['class'] = label_encoder.fit_transform(y_train['class'])
y_test['class'] = label_encoder.fit_transform(y_test['class'])
# print(y_train.columns)

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# model = Sequential([
#     Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
#     Dropout(0.5),
#     Dense(64, activation='relu'),
#     Dropout(0.5),
#     Dense(1, activation='sigmoid')
# ])

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer=RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])
model.fit(X_train, y_train, epochs=20, batch_size=128, validation_data=(X_test, y_test))
# model.fit(X_train, y_train, epochs=20, batch_size=128, validation_data=(X_test, y_test), callbacks=[early_stopping])


# Définir le nombre d'itérations pour l'affichage de la barre de progression
# n_iterations = 300
#
# # Initialiser tqdm avec le nombre total d'itérations
# with tqdm(total=n_iterations) as pbar:
#     # Boucle sur les itérations
#     for i in range(n_iterations):
#         # Entraîner le modèle pour une itération
#         model.set_params(warm_start=True)
#         model.fit(X_train, y_train)
#
#         # Mettre à jour la barre de progression
#         pbar.update(1)
#
#         # Calculer l'exactitude sur les données de validation ou de test si nécessaire
#         y_pred = model.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)
#         pbar.set_postfix(accuracy=accuracy)  # Mise à jour des informations affichées

# Une fois la boucle terminée, le modèle est entraîné


y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)

# Calculer la précision
precision = precision_score(y_test, y_pred)

# Calculer le rappel
recall = recall_score(y_test, y_pred)

# Calculer le score F1
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# y_pred = model.predict(X_test)
#
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))
# print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
