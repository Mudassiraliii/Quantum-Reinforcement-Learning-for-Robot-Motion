import tensorflow as tf
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from typing import List

# This Input_Layer is a standard Keras layer and needs no changes.
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
                self.tiled_keep = tf.tile(self.keep, multiples=[1, int(len(self.input_symbols) / self.n_input)])
                self.tiled_modify = tf.tile(self.modify, multiples=[1, int(len(self.input_symbols) / self.n_input)])
            else:
                self.tiled_keep = tf.repeat(self.keep, repeats=int(len(self.input_symbols) / self.n_input), axis=1)
                self.tiled_modify = tf.repeat(self.modify, repeats=int(len(self.input_symbols) / self.n_input), axis=1)
        self.input_parameters = self.add_weight("input_parameters", shape=(len(self.input_symbols),), initializer=tf.constant_initializer(1), dtype=tf.float32, trainable=trainable_input)

    def call(self, inputs):
        tensor_input = tf.convert_to_tensor(inputs, dtype=np.float32)
        if self.input_type == "data":
            tiled_input = tf.tile(tensor_input, multiples=[1, int(len(self.input_symbols) / self.n_input)])
        else:
            tiled_input = tf.repeat(tensor_input, repeats=int(len(self.input_symbols) / self.n_input), axis=1)
        if self.specific_training:
            modify_inputs = tf.math.multiply(tiled_input, self.tiled_modify)
            keep_inputs = tf.math.multiply(tiled_input, self.tiled_keep)
            scaled_modify_inputs = tf.einsum("i,ji->ji", self.input_parameters, modify_inputs)
            activated_modify_inputs = tf.keras.layers.Activation(self.activation)(scaled_modify_inputs)
            all_inputs = tf.math.add(keep_inputs, activated_modify_inputs)
        else:
            scaled_inputs = tf.einsum("i,ji->ji", self.input_parameters, tiled_input)
            all_inputs = tf.keras.layers.Activation(self.activation)(scaled_inputs)
        return all_inputs

# This is the final, correct version of the PQC_Qiskit layer.
class PQC_Qiskit(tf.keras.layers.Layer):
    def __init__(
        self,
        model_circuit: QuantumCircuit,
        operators: List[SparsePauliOp],
        input_symbols: list,
        circuit_symbols: list,
        **kwargs
    ):
        super().__init__(**kwargs)
        param_map = {p.name: p for p in model_circuit.parameters}
        input_params = [param_map[s.name] for s in input_symbols]
        circuit_params = [param_map[s.name] for s in circuit_symbols]
        self.qnn = EstimatorQNN(
            circuit=model_circuit,
            observables=operators,
            input_params=input_params,
            weight_params=circuit_params,
        )
        self.circuit_parameters = self.add_weight(
            "circuit_parameters",
            shape=(len(circuit_symbols),),
            initializer=tf.random_uniform_initializer(minval=0.0, maxval=2 * np.pi),
            dtype=tf.float32,
            trainable=True,
        )

    def call(self, inputs):

        def forward_pass(input_data, weight_data):
            qnn_forward_pass = self.qnn.forward(input_data, weight_data)
            return qnn_forward_pass.astype(np.float32)

        def backward_pass(input_data, weight_data):
            # Get the number of outputs from the length of the observables list
            num_outputs = len(self.qnn.observables)

            input_grad, weight_grad = self.qnn.backward(input_data, weight_data)

            if input_grad is None:
                # Use the correct num_outputs here
                input_grad = np.zeros((input_data.shape[0], num_outputs, self.qnn.num_inputs))
            if weight_grad is None:
                # Use the correct num_outputs here
                weight_grad = np.zeros((input_data.shape[0], num_outputs, self.qnn.num_weights))

            return input_grad.astype(np.float32), weight_grad.astype(np.float32)

        @tf.custom_gradient
        def custom_op(input_data, weight_data):
            output = tf.py_function(
                func=forward_pass,
                inp=[input_data, weight_data],
                Tout=tf.float32
            )

            def grad(dy):
                input_grad_np, weight_grad_np = tf.py_function(
                    func=backward_pass,
                    inp=[input_data, weight_data],
                    Tout=[tf.float32, tf.float32]
                )
                grad_inputs = tf.einsum('ij,ijk->ik', dy, input_grad_np)
                grad_weights = tf.einsum('ij,ijk->k', dy, weight_grad_np)
                return grad_inputs, grad_weights

            return output, grad

        output = custom_op(inputs, self.circuit_parameters)

        # Use the correct way to get the number of outputs here as well
        output.set_shape([None, len(self.qnn.observables)])

        return output

# This Output_Layer is a standard Keras layer and needs no changes.
class Output_Layer(tf.keras.layers.Layer):
    def __init__(self, units, rescale=False, trainable_output=True, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units) if not isinstance(units, int) else units
        self.rescale = rescale
        self.factors = self.add_weight("output_parameters", shape=(1, self.units), initializer="ones", trainable=trainable_output)

    def call(self, inputs):
        if self.rescale:
            return tf.math.multiply((inputs + 1) / 2, tf.repeat(self.factors, repeats=tf.shape(inputs)[0], axis=0))
        else:
            return tf.math.multiply(inputs, tf.repeat(self.factors, repeats=tf.shape(inputs)[0], axis=0))
