import numpy as np
import tensorflow as tf  # type: ignore
from qiskit_machine_learning.neural_networks import EstimatorQNN

class Input_Layer(tf.keras.layers.Layer):
    def __init__(
        self,
        input_symbols,
        n_input,
        activation=tf.math.atan,
        trainable_input=True,
        input_type="data",
        specific_training=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_symbols = input_symbols
        self.n_input = n_input
        self.activation = activation
        self.input_type = input_type
        self.specific_training = specific_training

        if specific_training:
            self.modify = tf.convert_to_tensor([[1, 1, 0]], dtype=np.float32)
            self.keep = tf.convert_to_tensor([[0, 0, 1]], dtype=np.float32)

            if self.input_type == "data":
                self.tiled_keep = tf.tile(
                    self.keep,
                    multiples=[1, int(len(self.input_symbols) / self.n_input)],
                )
                self.tiled_modify = tf.tile(
                    self.modify,
                    multiples=[1, int(len(self.input_symbols) / self.n_input)],
                )
            else:
                self.tiled_keep = tf.repeat(
                    self.keep,
                    repeats=int(len(self.input_symbols) / self.n_input),
                    axis=1,
                )
                self.tiled_modify = tf.repeat(
                    self.modify,
                    repeats=int(len(self.input_symbols) / self.n_input),
                    axis=1,
                )

        self.input_parameters = self.add_weight(
            "input_parameters",
            shape=(len(self.input_symbols),),
            initializer=tf.constant_initializer(1),
            dtype=tf.float32,
            trainable=trainable_input,
        )

    def call(self, inputs):
        tensor_input = tf.convert_to_tensor(inputs, dtype=np.float32)

        if self.input_type == "data":
            tiled_input = tf.tile(
                tensor_input, multiples=[1, int(len(self.input_symbols) / self.n_input)]
            )
        else:
            tiled_input = tf.repeat(
                tensor_input,
                repeats=int(len(self.input_symbols) / self.n_input),
                axis=1,
            )

        if self.specific_training:
            modify_inputs = tf.math.multiply(tiled_input, self.tiled_modify)
            keep_inputs = tf.math.multiply(tiled_input, self.tiled_keep)
            scaled_modify_inputs = tf.einsum(
                "i,ji->ji", self.input_parameters, modify_inputs
            )
            activated_modify_inputs = tf.keras.layers.Activation(self.activation)(
                scaled_modify_inputs
            )
            all_inputs = tf.math.add(keep_inputs, activated_modify_inputs)

        else:
            scaled_inputs = tf.einsum("i,ji->ji", self.input_parameters, tiled_input)
            all_inputs = tf.keras.layers.Activation(self.activation)(scaled_inputs)

        return all_inputs


class PQC_customized(tf.keras.layers.Layer):
    def __init__(
        self,
        model_circuit,
        input_symbols,
        circuit_symbols,
        operators,
        initializer=tf.keras.initializers.RandomUniform(0, np.pi),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.qnn = EstimatorQNN(
            circuit=model_circuit,
            observables=operators,
            input_params=input_symbols,
            weight_params=circuit_symbols,
            input_gradients=True,
        )
        self.circuit_parameters = self.add_weight(
            "circuit_parameters",
            shape=(len(circuit_symbols),),
            initializer=initializer,
            dtype=tf.float32,
            trainable=True,
        )

    def call(self, inputs):
        @tf.custom_gradient
        def forward_pass(in_data, weights):
            def _run_forward(i_d, w):
                return np.array(self.qnn.forward(i_d, w), dtype=np.float32)

            output = tf.numpy_function(_run_forward, [in_data, weights], tf.float32)
            output.set_shape((None, self.qnn.output_shape[0]))

            def backward_pass(upstream_grads):
                def _run_backward(in_d, w, g):
                    in_grad, w_grad = self.qnn.backward(in_d, w)

                    if in_grad is not None and in_grad.ndim == 3:
                         in_grad_tf = np.einsum("bj,bji->bi", g, in_grad)
                    elif in_grad is not None:
                         if in_grad.shape[0] == g.shape[0]: 
                              if g.shape[1] == 1:
                                  in_grad_tf = g * in_grad
                              else:
                                  in_grad_tf = in_grad 
                         else:
                              in_grad_tf = g @ in_grad
                    else:
                         in_grad_tf = np.zeros_like(in_d)

                    if w_grad is not None and w_grad.ndim == 3:
                         w_grad_tf = np.einsum("bj,bji->i", g, w_grad)
                    elif w_grad is not None and w_grad.ndim == 2:
                        if w_grad.shape[0] == g.shape[0]:
                             if g.shape[1] == 1:
                                  w_grad_tf = np.sum(g * w_grad, axis=0)
                             else:
                                  w_grad_tf = np.sum(w_grad, axis=0)
                        else:
                             w_grad_tf = np.einsum("bj,ji->i", g, w_grad)
                    else:
                         w_grad_tf = np.zeros_like(w)

                    return np.array(in_grad_tf, dtype=np.float32), np.array(
                        w_grad_tf, dtype=np.float32
                    )

                input_grads, weight_grads = tf.numpy_function(
                    _run_backward,
                    [in_data, weights, upstream_grads],
                    [tf.float32, tf.float32],
                )
                return input_grads, weight_grads

            return output, backward_pass

        return forward_pass(inputs, self.circuit_parameters)


class Output_Layer(tf.keras.layers.Layer):
    def __init__(self, units, rescale=False, trainable_output=True, **kwargs):

        super().__init__(**kwargs)
        self.units = int(units) if not isinstance(units, int) else units
        self.rescale = rescale

        self.factors = self.add_weight(
            "output_parameters",
            shape=(1, self.units),
            initializer="ones",
            trainable=trainable_output,
        )

    def call(self, inputs):
        if self.rescale:
            return tf.math.multiply(
                (inputs + 1) / 2,
                tf.repeat(self.factors, repeats=tf.shape(inputs)[0], axis=0),
            )
        else:
            return tf.math.multiply(
                inputs, tf.repeat(self.factors, repeats=tf.shape(inputs)[0], axis=0)
            )
