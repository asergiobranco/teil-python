from teil.ebis import _ClassifierBase

MLP_header_code = """
#ifndef __$MODELNAME$__
#define __$MODELNAME$__

#include "teil/config.h"
#include "teil/model/neural_network/mlpc.h" 

extern mlpc_model_t $MODELNAME$;
extern feature_type $MODELNAME$_sample[$NFEATURES$];
extern mlp_hidden_layer_t *$MODELNAME$_layers[$N_LAYERS$];

#endif
"""

hidden_layer_code = """
double $MODELNAME$_layer_$LAYER_NO$_weights[$N_PREV][$N_NEURONS$] = {
    {$WEIGHTS$}
};

double $MODELNAME$_layer_$LAYER_NO$_bias[$N_NEURONS$] = {
    $BIAS$
};

double $MODELNAME$_layer_$LAYER_NO$_neurons[$N_NEURONS$];

mlp_hidden_layer_t $MODELNAME$_layer_$LAYER_NO$ = {
    .weights = $MODELNAME$_layer_$LAYER_NO$_weights,
    .n_neurons = $N_NEURONS$,
    .bias = $MODELNAME$_layer_$LAYER_NO$_bias,
    .neurons = $MODELNAME$_layer_$LAYER_NO$_neurons,
    .activation = $ACTIVAtION$
};
"""

MLP_model_code = """

#include "teil/config.h"
#include "$MODELNAME$.h"
#include "teil/model/neural_network/mlpc.h" 
#include "teil/model/neural_network/neural_utils.h" 

$HIDDEN_LAYERS_CODE$

feature_type $MODELNAME$_sample[$NFEATURES$];
mlp_hidden_layer_t *$MODELNAME$_layers[$N_LAYERS$] = {$LAYERS$};

mlpc_model_t $MODELNAME$ = {
    .n_features = $NFEATURES$,
    .n_classes = $NCLASSES$,
    .n_layers = $N_LAYERS$,
    .sample = $MODELNAME$_sample,
    .hidden_layers = $MODELNAME$_layers
};

"""



class MLPClassifierTranspiler(_ClassifierBase):
    def __init__(self, model, model_name : str = "MLPC"):
        super().__init__(model, model_name, MLP_model_code, MLP_header_code )
    
    def _transpile(self):
        self.generate_layer_code()
        self._replace["$HIDDEN_LAYERS_CODE$"] = self._replace_hidden_layers()
        self._replace["$LAYERS$"] = self._replace_layers()
        self._replace["$N_LAYERS$"] = self._replace_n_layers()
        
    
    def _replace_n_layers(self):
        n_layers = len(self.model.intercepts_)
        return str(n_layers)

    def generate_layer_code(self):
        code = []
        i=0
        for weights, bias in zip(self.model.coefs_, self.model.intercepts_):
            aux = hidden_layer_code.replace("$N_NEURONS$", str(int(len(bias))))
            aux = aux.replace("$N_PREV", str(weights.shape[0]))
            aux = aux.replace("$MODELNAME$", self.model_name)
            aux = aux.replace("$LAYER_NO$", str(i))
            aux = aux.replace("$BIAS$", ','.join(bias.astype(str)))
            weights_code = '},\n\t{'.join(
                map(lambda x : ','.join(x), weights.astype(str))
            )
            aux = aux.replace("$WEIGHTS$", weights_code )
            i+=1
            if i == len(self.model.intercepts_):
               aux = aux.replace("$ACTIVAtION$", "ACTIVATION_IDENTITY")
            else:
                 aux = aux.replace("$ACTIVAtION$", "ACTIVATION_%s" % (self.model.activation.upper()))
            code.append(aux)
        self.hidden_layer_code = code
    

    def _replace_hidden_layers(self):
        return '\n'.join(self.hidden_layer_code)
    
    def _replace_layers(self):
        layers = ["&%s_layer_%d" % (self.model_name, x) for x in range(len(self.model.intercepts_))]
        return ','.join(layers)