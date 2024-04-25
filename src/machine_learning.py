
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



class ArffToDf:
    def __init__(self):
        self.arff_file_train = arff.loadarff(coarff.PATH_ARFF_TRAIN.value)
        self.arff_file_test = arff.loadarff(coarff.PATH_ARFF_TEST.value)

    def dataframe(self):
        df_train = pd.DataFrame(self.arff_file_train[0])
        df_test = pd.DataFrame(self.arff_file_test[0])
        return df_train,df_test


df_train,df_test = ArffToDf().dataframe()
# print(df_test)
# print(type(df_train))


X_train = df_train.iloc[:, :-1]
X_test = df_test.iloc[:, :-1]

derniere_colonne_train = df_train.iloc[:, -1]
derniere_colonne_test = df_test.iloc[:, -1]
y_train = pd.DataFrame(derniere_colonne_train, columns=['class'])
y_test = pd.DataFrame(derniere_colonne_test, columns=['class'])

print(X_train.shape)
print(X_test.shape)

# Appliquer l'encodage one-hot sur la colonne catégorielle
X_train = pd.get_dummies(X_train, columns=["protocol_type","service","flag"])
X_test = pd.get_dummies(X_test, columns=["protocol_type","service","flag"])
# y_train = pd.get_dummies(y_train, columns=["class"])
# y_test = pd.get_dummies(y_test, columns=["class"])

# Trouver les colonnes manquantes dans l'ensemble de test
missing_columns = set(X_train.columns) - set(X_test.columns)
print(missing_columns)

for column in missing_columns:
    X_test[column] = 0

X_test = X_test[X_train.columns]

print(X_train.shape)
print(X_test.shape)
print(X_train["service_b'aol'"])
print(X_test["service_b'aol'"])

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
    Dense(8, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

# model = Sequential([
#     Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
#     Dropout(0.5),
#     Dense(64, activation='relu'),
#     Dropout(0.5),
#     Dense(1, activation='sigmoid')
# ])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])
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
