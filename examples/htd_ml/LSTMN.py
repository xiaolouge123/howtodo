from keras.layers.recurrent import _generate_dropout_mask, LSTM, LSTMCell, RNN
from keras.engine.base_layer import Layer
import keras.backend as K
from keras.initializers import glorot_normal, orthogonal, zero
import keras.regularizers as regularizers
import keras


class LSTMN(Layer):
    """
    Paper: https://arxiv.org/pdf/1601.06733.pdf
    Code inspired by https://github.com/oneil512/lstmn/blob/master/lstmn.py
    """
    def __init__(self,units,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=2,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        
        self.cell = LSTMCell(units,
                activation=activation,
                recurrent_activation=recurrent_activation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                recurrent_initializer=recurrent_initializer,
                unit_forget_bias=unit_forget_bias,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                recurrent_regularizer=recurrent_regularizer,
                bias_regularizer=bias_regularizer,
                kernel_constraint=kernel_constraint,
                recurrent_constraint=recurrent_constraint,
                bias_constraint=bias_constraint,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                implementation=implementation)
        
        # super(LSTMN, self).__init__(cell,
        #         return_sequences=return_sequences,
        #         return_state=return_state,
        #         go_backwards=go_backwards,
        #         stateful=stateful,
        #         unroll=unroll,
        #         **kwargs)
        att_size = 128
        self.v = K.ones(shape=(1,att_size))
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.name = 'LSTMN'
    
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.batch_size = 5
        self.steps = input_shape[1]
        self.dim = input_shape[-1]

        
        self.kernel_att_x = self.add_weight(shape=(input_dim, 128),
                                            name='attention_x_kernel',
                                            initializer=self.kernel_initializer,
                                            regularizer=None,
                                            constraint=None)
        
        self.kernel_att_h = self.add_weight(shape=(128, 128),
                                            name='kernel_att_h',
                                            initializer=self.kernel_initializer,
                                            regularizer=None,
                                            constraint=None)
        
        self.kernel_att_h_tilde = self.add_weight(shape=(128, 128),
                                            name='kernel_att_h_tilde',
                                            initializer=self.kernel_initializer,
                                            regularizer=None,
                                            constraint=None)

    def call(self, inputs):
        # wrapper around lstmcell
        batch_hidden_states = []
        outputs = []

        for i in range(self.batch_size):
            x = inputs[i]

            hidden_states = []
            past_ht = []
            past_x = []
            past_c = []
            
            for step in range(self.steps):
                att_vec = []
                past_x.append(x[step])
                
                for k, h in enumerate(hidden_states):
                    att_k = K.dot(
                        self.v,
                        K.tanh(K.dot(self.kernel_att_h, h) + K.dot(self.kernel_att_x, x) + K.dot(self.kernel_att_h_tilde, past_ht[step]))
                    )
                    att_vec.append(att_k[0][0])
                att_softmax = K.softmax(K.variable(att_vec))

                self.ht_tilde = K.zeros(shape=(1,128))
                self.ct_tilde = K.zeros(shape=(1,128))
                if len(hidden_states) > 0:
                    for k, s in enumerate(att_softmax):
                        self.ht_tilde += s * hidden_states[k]
                        self.ct_tilde += s * past_c[k]
                    

                h, h_c  = self.cell.call(x[step], [self.ht_tilde, self.ct_tilde])
                self.ht = h
                self.ct = h_c[-1]

                past_ht.append(self.ht_tilde)
                hidden_states.append(self.ht)
                past_c.append(self.ct)
            
            batch_hidden_states.append(hidden_states)
            outputs.append(self.ht)
        return outputs, batch_hidden_states


def build_model():
    inputs = keras.Input(shape=(5,), dtype="int32")
    x = keras.layers.Embedding(20000, 128)(inputs)
    x = LSTMN(128)(x)
    outputs = keras.layers.Dense(3, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.summary()

if __name__=="__main__":
    build_model()


