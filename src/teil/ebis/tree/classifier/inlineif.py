from sklearn.tree import DecisionTreeClassifier
import numpy as np 

from teil.ebis import _ClassifierBase

MODEL_H_CODE = """
#ifndef __$MODELNAME$__
#define __$MODELNAME$__

#include "teil/model/tree/tree.h"

#define $MODELNAME$_PROBABILITY 1

extern dt_model_t $MODELNAME$;
extern float $MODELNAME$_probability[$NCLASSES$];
extern feature_type $MODELNAME$_sample[$NFEATURES$];
extern output_node_t $MODELNAME$_out;

/*
Use the following function to pedict.
Example:
    #include "$MODELNAME$.h"
    $MODELNAME$_out = $MODELNAME$_predict_if_DT(
        $MODELNAME$, 
        $MODELNAME$_out
    );

    $MODELNAME$_out->output; // class int

    // Array containing the probability for each class
    // Only available if $MODELNAME$_PROBABILITY defined
    $MODELNAME$_out->probability; 
*/

output_node_t * $MODELNAME$_predict_if_DT(dt_model_t model, output_node_t * output);

#endif 
"""

MODEL_C_CODE = """
#include "teil/model/tree/tree.h"
#include "teil/config.h"
#include "$MODELNAME$.h"
#include <stddef.h>



float $MODELNAME$_probability[$NCLASSES$] = {0.0};

output_node_t $MODELNAME$_out = {
    .output = 0,
    .probability = $MODELNAME$_probability
};
feature_type $MODELNAME$_sample[$NFEATURES$] = {0.0};

dt_model_t $MODELNAME$ = {
    .sample = $MODELNAME$_sample,
    .n_features = $NFEATURES$,
    .n_classes = $NCLASSES$,
    .root_node = NULL
};

output_node_t * $MODELNAME$_predict_if_DT(dt_model_t model, output_node_t * output){
$IFCODE$
}
"""

class _DTC_inline_if(_ClassifierBase):
    def __init__(self, model : DecisionTreeClassifier, model_name : str = "DT"):
        super().__init__(
            model, 
            model_name,
            MODEL_C_CODE,
             MODEL_H_CODE
        )

    def _transpile(self):
        self._replace["$IFCODE$"] = self.build_inline_if()
    
    def build_inline_if(self):
        if_text = self._build_inline_if(0, 2)
        return if_text
    
    def _build_inline_if(self, i, tab=0):
        if(self.model.tree_.feature[i] == -2):
            tab = "   "*tab
            code = ""
            code += "{tab}output->output = {classe};\n{tab}#ifdef $MODELNAME$_PROBABILITY\n".format( 
                tab = tab, 
                classe = np.argmax(self.model.tree_.value[i][0])
            )
            for j, proba in enumerate(self.model.tree_.value[i][0]):
                code += "{tab}output->probability[{i}] = {proba_val};\n".format(tab=tab, i=j, proba_val = proba)
            code += "{tab}#endif\n".format(tab=tab)
            return code
            #return "%sreturn %s;" % ("  "*tab, "end_nodes[%d]" % (np.where(self.end_nodes == i)[0][0]))
        else:
            return "{tab}if(sample[{feture_id}] < {threshold}){{\n\r{left}\n\r{tab}}}\n\r{tab}else{{{tab}\n\r{right}\n\r{tab}}}" .format (
                tab="   "*tab,
                feture_id=self.model.tree_.feature[i],
                threshold=self.model.tree_.threshold[i],
                left=self._build_inline_if( self.model.tree_.children_left[i], tab=tab+1),
                right=self._build_inline_if( self.model.tree_.children_right[i], tab=tab+1)
            )