from teil.ebis.neural_network import MLPRegressorTranspiler
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import load_diabetes

dt = MLPRegressor()
X, y = load_diabetes(return_X_y=True)

dt.fit(X, y)

transpiler = MLPRegressorTranspiler(
    model = dt,
    model_name = "MLPRdiabetes"
)

transpiler.transpile(
    folder_path = "./",
    save_to_file=True
)