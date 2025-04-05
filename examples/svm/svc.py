from sklearn.datasets import load_iris
from sklearn.svm import SVC
from teil.ebis.svm import SVCTranspiler

X, y = load_iris(
    return_X_y=True
)

svc_linear = SVC(kernel="linear")

svc_linear.fit(X, y)

transpiler = SVCTranspiler(
    model = svc_linear,
    model_name = "SvcIrisLinear"
)

transpiler.transpile(
    save_to_file = True,
    folder_path = "./"
)

svc_linear = SVC(kernel="poly")

svc_linear.fit(X, y)

transpiler = SVCTranspiler(
    model = svc_linear,
    model_name = "SvcIrisPoly"
)

transpiler.transpile(
    save_to_file = True,
    folder_path = "./"
)

svc_linear = SVC(kernel="rbf")

svc_linear.fit(X, y)

transpiler = SVCTranspiler(
    model = svc_linear,
    model_name = "SvcIrisRBF"
)

transpiler.transpile(
    save_to_file = True,
    folder_path = "./"
)