from teil.ebis.neural_network import MLPClassifierTranspiler
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris

dt = MLPClassifier()
X, y = load_iris(return_X_y=True)

dt.fit(X, y)

transpiler = MLPClassifierTranspiler(
    model = dt,
    model_name = "MLPCIris"
)

transpiler.transpile(
    folder_path = "./",
    save_to_file=True
)