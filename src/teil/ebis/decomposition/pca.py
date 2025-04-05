import numpy as np 

from teil.ebis import _RegressorBase

PCAc_code = """
#include <stddef.h>

#include "teil/model/pca/pca.h"
#include "teil/config.h"

#include "$MODELNAME$.h"

double $MODELNAME$_pcs[$N_PCS$][$NFEATURES$] = {
    {$PRINCIPAL_COMPONENTS$}
};
double $MODELNAME$_sv[$N_PCS$] = {$SINGULAR_VALUES$};
double $MODELNAME$_ratio[$N_PCS$] = {$RATIO$};

feature_type $MODELNAME$_sample[$NFEATURES$] = {0.0};
double $MODELNAME$_inverse[$NFEATURES$] = {0.0};
double $MODELNAME$_error[$NFEATURES$] = {0.0};
double $MODELNAME$_scores[$N_PCS$] = {0.0};

pca_model_t $MODELNAME$  = {
    .sample = $MODELNAME$_sample,
    .n_features = $NFEATURES$,
    .n_pcs = $N_PCS$,
    .correlation_matrix = NULL,
    .U = NULL,
    .v = $MODELNAME$_pcs,
    .singular_values = $MODELNAME$_sv,
    .explained_variance_ratio = $MODELNAME$_ratio, 
    .scores = $MODELNAME$_scores,
    .inverse = $MODELNAME$_inverse,
    .error = $MODELNAME$_error
};
"""

PCAheader_code = """
#ifndef __$MODELNAME$__
#define __$MODELNAME$__

#include "teil/model/pca/pca.h"
#include "teil/config.h"


extern pca_model_t $MODELNAME$;

extern double $MODELNAME$_pcs[$N_PCS$][$NFEATURES$];
extern double $MODELNAME$_sv[$N_PCS$];
extern double $MODELNAME$_ratio[$N_PCS$];

extern feature_type $MODELNAME$_sample[$NFEATURES$];
extern double $MODELNAME$_inverse[$NFEATURES$];
extern double $MODELNAME$_error[$NFEATURES$];
extern double $MODELNAME$_scores[$N_PCS$];

#endif
"""

class PCATranspiler(_RegressorBase):
    def __init__(self, model, model_name : str = "PCA"):
        super().__init__(
            model, 
            model_name,
            PCAc_code,
            PCAheader_code
        )

    def _transpile(self):
        self.r = {
            "$N_PCS$" : self._replace_n_pcs(),
            "$PRINCIPAL_COMPONENTS$" : self._replace_components(),
            "$RATIO$" : self._replace_ratio(),
            "$SINGULAR_VALUES$" : self._replace_sv()
        } 
        self._replace = {**self._replace , **self.r}
    
    def _replace_sv(self):
        return ",".join(self.model.explained_variance_.astype(str))
    
    def _replace_components(self):
        return '},\n\t{'.join(
            map(lambda x : ','.join(x), self.model.components_.astype(str))
        )
       
    def _replace_ratio(self):
        return ",".join(self.model.explained_variance_ratio_.astype(str))
    
    def _replace_n_pcs(self):
        return str(self.model.n_components_)