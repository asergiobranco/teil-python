from teil.ebis.decomposition import PCATranspiler as Transpiler
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

dt = PCA(n_components=1)
X, y = load_iris(return_X_y=True)

dt.fit(X, y)

transpiler = Transpiler(
    model = dt,
    model_name = "PCAIris"
)

transpiler.transpile(
    folder_path = "./",
    save_to_file=True
)