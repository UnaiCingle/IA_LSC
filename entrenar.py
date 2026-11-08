# Importar les llibreries
import pandas as pd

import joblib

from sklearn.neural_network import MLPClassifier

# CSV
data = pd.read_csv("data/dataset.csv", header=None)

y = data.iloc[:, 0]
X = data.iloc[:, 1:]

model = MLPClassifier(
    hidden_layer_sizes=(32,),
    max_iter=10000
)

# Entrenar
print("Entrenant...")
model.fit(X, y)
print("Llest!")

joblib.dump(model, "models/lsc.pkl")
