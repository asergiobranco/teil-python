from teil.ebis.tree.classifier import _DTC_inline_if
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

dt = DecisionTreeClassifier()
X, y = load_iris(return_X_y=True)

dt.fit(X, y)

transpiler = _DTC_inline_if(
    model = dt,
    model_name="DTCIris"
)

transpiler.transpile(
    save_to_file = True,
    folder_path = "./"
)