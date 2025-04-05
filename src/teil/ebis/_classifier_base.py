class _ClassifierBase:
    def __init__(self, model, model_name : str, c_code : str, h_code : str):
        self.model = model
        self.model_name = model_name
        self.n_features = model.n_features_in_
        if hasattr(model, "n_classes_"):
            self.n_classes = model.n_classes_
        else:
            self.n_classes = len(model.classes_)
        self.code = c_code
        self.header = h_code
        self._replace = {
            "$NCLASSES$" : str(self.n_classes),
            "$NFEATURES$" : str(self.n_features),
            "$MODELNAME$" : self.model_name
        }
    
    def _replace_in_code(self):
        """Replaces the variables ($VAR_NAME$) in the headers and code 
        template."""
        for _ in range(2):
            for key, value in self._replace.items():
                self.code = self.code.replace(
                    key, 
                    value
                )
                self.header = self.header.replace(
                    key, 
                    value
                )

    def transpile(
        self,
        save_to_file=False,
        folder_path="./"
    ):
        if not folder_path.endswith("/"):
            folder_path += "/"

        self._transpile()
        
        self._replace_in_code()

        if save_to_file:
            with open(folder_path + self.model_name + ".c", "w+") as fp:
                fp.write(self.code)
            
            with open(folder_path + self.model_name + ".h", "w+") as fp:
                fp.write(self.header)
