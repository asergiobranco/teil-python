from sklearn.datasets import load_diabetes
from sklearn.svm import SVR
from teil.ebis.svm import SVRTranspiler

X, y = load_diabetes(
    return_X_y=True
)

SVR_linear = SVR(kernel="linear")

SVR_linear.fit(X, y)

transpiler = SVRTranspiler(
    model = SVR_linear,
    model_name = "SVRdiabetesLinear"
)

transpiler.transpile(
    save_to_file = True,
    folder_path="/home/teiluser/workspace/"
)

SVR_linear = SVR(kernel="poly")

SVR_linear.fit(X, y)

transpiler = SVRTranspiler(
    model = SVR_linear,
    model_name = "SVRdiabetesPoly"
)

transpiler.transpile(
    save_to_file = True,
    folder_path="/home/teiluser/workspace/"
)

SVR_linear = SVR(kernel="rbf")

SVR_linear.fit(X, y)

transpiler = SVRTranspiler(
    model = SVR_linear,
    model_name = "SVRdiabetesRBF"
)

transpiler.transpile(
    save_to_file = True,
    folder_path="/home/teiluser/workspace/"
)