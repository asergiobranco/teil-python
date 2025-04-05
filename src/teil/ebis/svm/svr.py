import numpy as np 
import os 

from teil.ebis import _RegressorBase


SVClinear_c_code = """
#include "teil/model.h"
#include "teil/config.h"
#include "teil/model/svm/svr.h"
#include "teil/model/preprocessing/kernel.h"
#include <stddef.h>

feature_type $MODELNAME$_sample[$NFEATURES$];

feature_type $MODELNAME$_sv[$NFEATURES$] = {
    $SUPPORT_VECTORS$
};

svr_model_t $MODELNAME$ = {
    .n_support = 1,
    .n_features = $NFEATURES$,
    .degree = 0,
    .gamma = 0.0,
    .coef = 0.0,
    .kernel = KERNEL_LINEAR,
    .kernels = NULL,
    .support_vectors =  NULL,
    .weights = $MODELNAME$_sv,
    .intercepts = $INTERCEPTS$,
    .sample = $MODELNAME$_sample,
    .decision_rules = 0,
};

"""

SVClinear_h_code = """
#ifndef __$MODELNAME$__
#define __$MODELNAME$__

#include "teil/model/svm/svr.h"

extern svr_model_t $MODELNAME$;

#endif
"""

SVC_c_code = """
#include "teil/model.h"
#include "teil/config.h"
#include "teil/model/svm/svr.h"
#include "teil/model/preprocessing/kernel.h"
#include <stddef.h>

feature_type $MODELNAME$_sample[$NFEATURES$];
feature_type $MODELNAME$_kernels[$N_SVS$];
feature_type $MODELNAME$_sv[$N_SVS$][$NFEATURES$] = {
    {$SUPPORT_VECTORS$}
};

feature_type $MODELNAME$_weights[$N_SVS$] = {
    $WEIGHTS$
};

svr_model_t $MODELNAME$ = {
    .n_support = $N_SVS$,
    .n_features = $NFEATURES$,
    .degree = $DEGREE$,
    .gamma = $GAMMA$,
    .coef = $COEF$,
    .kernel = KERNEL_$KERNEL$,
    .kernels = $MODELNAME$_kernels,
    .support_vectors =  $MODELNAME$_sv,
    .weights = $MODELNAME$_weights,
    .intercepts = $INTERCEPTS$,
    .sample = $MODELNAME$_sample,
    .decision_rules = 0.0,
};

"""

class SVRTranspiler(_RegressorBase):
    def __init__(self, model, model_name : str = "SVR"):
        if model.kernel == "linear":
            super().__init__(model, model_name, SVClinear_c_code, SVClinear_h_code)
        else:
            super().__init__(model, model_name, SVC_c_code, SVClinear_h_code)

    def _transpile(self):
        self._replace["$SUPPORT_VECTORS$"] = self._replace_support_vectors()
        self._replace["$INTERCEPTS$"] = self._replace_intercepts()
        self._replace["$N_SVS$"] =self._replace_n_svs()
        self._replace["$WEIGHTS$"] = self._replace_weights()
        self._replace["$DEGREE$"] = self._replace_degree()
        self._replace["$GAMMA$"] = self._replace_gamma()
        self._replace["$COEF$"] = self._replace_coef()
        self._replace["$KERNEL$"] = self._replace_kernel()


    def _replace_n_svs(self):
        return str(len(self.model.support_vectors_))
    
    def _replace_degree(self):
        if self.model.kernel == "linear":
            return ""
        else:
            return str(self.model.degree)

    def _replace_gamma(self):
        if self.model.kernel == "linear":
            return ""
        else:
            return str(self.model._gamma)

    def _replace_kernel(self):
        if self.model.kernel == "linear":
            return ""
        else:
            return self.model.kernel.upper()

    def _replace_coef(self):
        if self.model.kernel == "linear":
            return ""
        else:
            return str(self.model.coef0)

    def _replace_n_intercepts(self):
        return str((self.n_classes * (self.n_classes-1)) // 2)


    def _replace_intercepts(self):
        prior_code = ",".join(self.model.intercept_.astype(str))
        return prior_code

    def _replace_weights(self):
        if self.model.kernel == "linear":
            return ""
        else:
            var_code = '},\n\t{'.join(
                map(lambda x : ','.join(x), self.model.dual_coef_.astype(str))
            )
            return var_code
    
    def _replace_ranges(self):
        if self.model.kernel == "linear":
            return ""
        else:
            ranges = [0]
            for i, v in enumerate(self.model.n_support_):
                ranges.append(ranges[-1] + v)

            return ','.join(map(lambda x: str(x), ranges))

    def _replace_support_vectors(self):
        if self.model.kernel == "linear":
            var_code = '},\n\t{'.join(
                map(lambda x : ','.join(x), self.model.coef_.astype(str))
            )
            return var_code
        else:
            var_code = '},\n\t{'.join(
                map(lambda x : ','.join(x), self.model.support_vectors_.astype(str))
            )
            return var_code