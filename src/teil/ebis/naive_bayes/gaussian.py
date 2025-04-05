import numpy as np 
import os 

from teil.ebis import _ClassifierBase

c_code = """
#include "teil/models/bayes/gaussiannb.h"
#include "teil/config.h"
#include "$MODELNAME$.h"

feature_type $MODELNAME$_sample[$NFEATURES$];
feature_type $MODELNAME$_pre_computed[$NCLASSES$][$NFEATURES$] = {
    {$PRECOMPUTED_CODE$}
};
feature_type $MODELNAME$_var_[$NCLASSES$][$NFEATURES$] = {
    {$VAR_CODE$}
};
feature_type $MODELNAME$_theta_[$NCLASSES$][$NFEATURES$] = {
    {$THETA_CODE$}
};
feature_type $MODELNAME$_class_prior[$NCLASSES$] = {$CLASS_PRIOR_CODE$};
feature_type $MODELNAME$_joint_log_likelihood[$NCLASSES$] = {0.0};

gaussian_nb_model_t $MODELNAME$ = {
    .n_features = $NFEATURES$,
    .n_classes = $NCLASSES$,
    .sample = $MODELNAME$_sample,
    .class_prior = $MODELNAME$_class_prior,
    .joint_log_likelihood = $MODELNAME$_joint_log_likelihood,
    .pre_computed = $MODELNAME$_pre_computed,
    .var_ = $MODELNAME$_var_,
    .theta_ = $MODELNAME$_theta_

};

"""

header_code = """
#ifndef __$MODELNAME$__
#define __$MODELNAME$__

#include "teil/models/bayes/gaussiannb.h"

enum {
    $CLASS_ENUM$
} $MODELNAME$_classes;

extern gaussian_nb_model_t $MODELNAME$;
extern feature_type $MODELNAME$_sample[$NFEATURES$];

#endif
"""



class GaussianNBTranspiler(_ClassifierBase):
    def __init__(self, model, model_name : str = "GaussianNB"):
        super().__init__(model, model_name, c_code, header_code)
        
    
    def _transpile(self):
        self._replace["$CLASS_PRIOR_CODE$"] = self._replace_class_prior()
        self._replace["$VAR_CODE$"] =  self._replace_var()
        self._replace["$THETA_CODE$"] =  self._replace_theta()
        self._replace["$PRECOMPUTED_CODE$"] =  self._replace_precomputed() 
        self._replace["$CLASS_ENUM$"] =  self._replace_class_enum()
        
    
    def _replace_class_prior(self):
        prior_code = ",".join(np.log(self.model.class_prior_).astype(str))
        self.c_code = self.c_code.replace("$CLASS_PRIOR_CODE$", prior_code)
    
    def _replace_var(self):
        var_code = '},\n\t{'.join(
            map(lambda x : ','.join(x), self.model.var_.astype(str))
        )
        self.c_code = self.c_code.replace("$VAR_CODE$", var_code)

    def _replace_theta(self):
        theta_code = '},\n\t{'.join(
            map(lambda x : ','.join(x), self.model.theta_.astype(str))
        )
        self.c_code = self.c_code.replace("$THETA_CODE$", theta_code)
    
    def _replace_precomputed(self):
        pre_computed_code = '},\n\t{'.join(
            map(lambda x : ','.join(x), np.log(2.0 * np.pi * self.model.var_).astype(str))
        )
        self.c_code = self.c_code.replace("$PRECOMPUTED_CODE$", pre_computed_code)
    
    def _replace_class_enum(self):
        t = []
        for cls in self.model.classes_:
            if str(cls).isnumeric():
                t.append("cls_%s" % (int(cls)))
            else:
                t.append(cls)
        self.header_code = self.header_code.replace("$CLASS_ENUM$", ",\n\t".join(t))
    