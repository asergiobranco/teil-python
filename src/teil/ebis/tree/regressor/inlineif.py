from sklearn.tree import DecisionTreeRegressor
import numpy as np 

from teil.ebis import _RegressorBase

MODEL_H_CODE = """

#ifndef __$MODELNAME$__
#define __$MODELNAME$__

#include "teil/model/tree/dtr.h"


extern dtr_model_t $MODELNAME$;
extern feature_type $MODELNAME$_sample[$NFEATURES$];
extern dtr_output_node_t $MODELNAME$_out;

/*
Use the following function to pedict.
Example:
    #include "$MODELNAME$.h"
    $MODELNAME$_out = $MODELNAME$_predict_if_dtr(
        $MODELNAME$, 
        $MODELNAME$_out
    );

    $MODELNAME$_out->output; // float
*/

dtr_output_node_t * $MODELNAME$_predict_if_dtr(dtr_model_t model, dtr_output_node_t * output);

#endif
"""

MODEL_C_CODE = """
#include "teil/model/tree/dtr.h"
#include "teil/config.h"
#include "$MODELNAME$.h"
#include <stddef.h>




dtr_output_node_t $MODELNAME$_out = {
    .output = 0,
};

feature_type $MODELNAME$_sample[$NFEATURES$] = {0.0};

dtr_model_t $MODELNAME$ = {
    .sample = $MODELNAME$_sample,
    .n_features = $NFEATURES$,
    .root_node = NULL
};

dtr_output_node_t* $MODELNAME$_predict_if_dtr(dtr_model_t model, dtr_output_node_t * output){
$IFCODE$
}
"""

class _DTC_inline_if(RegressorBase):
    def __init__(self, model : DecisionTreeRegressor, model_name : str = "DTR"):
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
            code += "{tab}output->output = {classe};\n{tab}".format( 
                tab = tab, 
                classe = self.model.tree_.value[i][0]
            )
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