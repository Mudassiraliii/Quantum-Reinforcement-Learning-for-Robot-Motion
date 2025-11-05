import numpy as np
import tensorflow as tf  # type: ignore

# --- New Qiskit Imports ---
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import Parameter
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TensorFlowConnector
# --- End New Imports ---


class Input_Layer(tf.keras.layers.Layer):
    """
    This class is unchanged. It's pure TensorFlow.
    """
    def __init__(
        self,
        input_symbols,  # In Step 5, we'll see this is now a list[Parameter]
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
    """
    --- THIS IS THE NEW, REWRITTEN CLASS ---
    It replaces the old PQC_customized and removes all tfq logic.
    """
    def __init__(
        self,
        model_circuit: QuantumCircuit,
        input_symbols: list[Parameter],
        circuit_symbols: list[Parameter],
        operators: list[SparsePauliOp],
        initializer=tf.keras.initializers.RandomUniform(0, np.pi),
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # 1. Create the EstimatorQNN (the "brain")
        self.qnn = EstimatorQNN(
            circuit=model_circuit,
            observables=operators,
            input_params=input_symbols,
            weight_params=circuit_symbols,
        )

        # 2. Set the initial weights for the trainable circuit parameters
        initial_weights = initializer(shape=(len(circuit_symbols),))

        # 3. Create the TensorFlowConnector (the Keras Layer)
        # This connector will automatically manage the circuit's trainable
        # weights (`circuit_symbols`) as Keras trainable variables.
        self.computation_layer = TensorFlowConnector(
            self.qnn,
            initial_weights=initial_weights
        )

    def call(self, inputs):
        """
        The new call method is extremely simple.
        We just pass the inputs from the Input_Layer
        directly to the TensorFlowConnector.
        """
        return self.computation_layer(inputs)


class Output_Layer(tf.keras.layers.Layer):
    """
    This class is also unchanged. It's pure TensorFlow.
    """
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
