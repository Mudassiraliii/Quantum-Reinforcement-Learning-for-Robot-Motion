from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


def get_new_param(symbol_name, qubit, position, layer=None):
    """
    Return a new learnable Qiskit parameter.
    """
    if layer is not None:
        name = f"{symbol_name}_{qubit}_{layer}_{position}"
    else:
        name = f"{symbol_name}_{qubit}_{position}"
    return Parameter(name)


def create_input(n_qubit, symbol_name, layer=None, input_style=1):
    """
    Creates a 1-qubit circuit for data input.
    n_qubit is the qubit's index, used for parameter naming.
    """
    circuit = QuantumCircuit(1)  # Circuit for 1 qubit
    input_parameters = []

    if input_style == 1:
        input_parameter = get_new_param(symbol_name, n_qubit, 0, layer)
        circuit.rx(input_parameter, 0)
        input_parameters.append(input_parameter)

    elif input_style == 2:
        input_parameter_0 = get_new_param(symbol_name, n_qubit, 0, layer)
        input_parameter_1 = get_new_param(symbol_name, n_qubit, 1, layer)
        
        circuit.rx(input_parameter_0, 0)
        circuit.ry(input_parameter_1, 0)
        
        input_parameters.append(input_parameter_0)
        input_parameters.append(input_parameter_1)

    elif input_style == 3:
        input_parameter_0 = get_new_param(symbol_name, n_qubit, 0, layer)
        input_parameter_1 = get_new_param(symbol_name, n_qubit, 1, layer)
        input_parameter_2 = get_new_param(symbol_name, n_qubit, 2, layer)
        
        circuit.rx(input_parameter_0, 0)
        circuit.ry(input_parameter_1, 0)
        circuit.rx(input_parameter_2, 0)  # Matches original rx, ry, rx
        
        input_parameters.append(input_parameter_0)
        input_parameters.append(input_parameter_1)
        input_parameters.append(input_parameter_2)
    else:
        raise ValueError(f"input_style {input_style} not implemented.")

    return circuit, input_parameters


def create_entanglement(num_qubits):
    """
    Creates the (circular) CZ entanglement layer for all qubits.
    """
    circuit = QuantumCircuit(num_qubits)

    for i in range(num_qubits - 1):
        circuit.cz(i, i + 1)
    
    if num_qubits > 2:
        circuit.cz(num_qubits - 1, 0)  # Circular entanglement

    return circuit


def create_unitary(n_qubit, layer, rot_per_unitary=3, circuit_type="skolik"):
    """
    Creates a 1-qubit trainable unitary.
    n_qubit is the qubit's index, used for parameter naming.
    """
    circuit = QuantumCircuit(1)
    body_parameters = []
    symbol_name = "train"

    if rot_per_unitary == 2:
        body_parameter_0 = get_new_param(symbol_name, n_qubit, 0, layer)
        body_parameter_1 = get_new_param(symbol_name, n_qubit, 1, layer)
        
        circuit.ry(body_parameter_0, 0)
        circuit.rz(body_parameter_1, 0)
        
        body_parameters.append(body_parameter_0)
        body_parameters.append(body_parameter_1)

    elif rot_per_unitary == 3 and circuit_type == "skolik":
        body_parameter_0 = get_new_param(symbol_name, n_qubit, 0, layer)
        body_parameter_1 = get_new_param(symbol_name, n_qubit, 1, layer)
        body_parameter_2 = get_new_param(symbol_name, n_qubit, 2, layer)
        
        circuit.rx(body_parameter_0, 0)
        circuit.ry(body_parameter_1, 0)
        circuit.rz(body_parameter_2, 0)
        
        body_parameters.append(body_parameter_0)
        body_parameters.append(body_parameter_1)
        body_parameters.append(body_parameter_2)

    elif rot_per_unitary == 3 and circuit_type == "han":
        body_parameter_0 = get_new_param(symbol_name, n_qubit, 0, layer)
        body_parameter_1 = get_new_param(symbol_name, n_qubit, 1, layer)
        body_parameter_2 = get_new_param(symbol_name, n_qubit, 2, layer)
        
        circuit.rx(body_parameter_0, 0)
        circuit.ry(body_parameter_1, 0)
        circuit.rx(body_parameter_2, 0)
        
        body_parameters.append(body_parameter_0)
        body_parameters.append(body_parameter_1)
        body_parameters.append(body_parameter_2)

    else:
        raise ValueError("rot_per_unitary =/= (2,3) not implemented")

    return circuit, body_parameters


def create_reupload_circuit(
    num_qubits,
    layers,
    rot_per_unitary=3,
    input_style=3,
    data_reupload=True,
    trainable_input=True,
    zero_layer=True,
):

    circuit = QuantumCircuit(num_qubits)

    input_symbols_name = "in"
    if trainable_input:
        input_symbols_name = "in_train"

    input_symbols = []
    trainable_symbols = []

    for layer in range(layers + 1):

        if layer == 0 and zero_layer:
            # === Initial Trainable Layer ===
            layer_circuit = QuantumCircuit(num_qubits)
            for i in range(num_qubits):
                circuit_unitary, unitary_parameters = create_unitary(
                    i, layer, rot_per_unitary
                )
                # Add the 1-qubit unitary to the correct qubit (i)
                layer_circuit.compose(circuit_unitary, [i], inplace=True)
                trainable_symbols.extend(unitary_parameters)
            
            circuit.compose(layer_circuit, inplace=True)
            circuit.compose(create_entanglement(num_qubits), inplace=True)

        elif layer > 0:
            # === Data Re-upload Layer ===
            if layer == 1 or data_reupload:
                input_layer_circuit = QuantumCircuit(num_qubits)
                for i in range(num_qubits):
                    input_circuit, input_parameters = create_input(
                        i, input_symbols_name, layer, input_style
                    )
                    # Add the 1-qubit input circuit to the correct qubit (i)
                    input_layer_circuit.compose(input_circuit, [i], inplace=True)
                    input_symbols.extend(input_parameters)
                
                circuit.compose(input_layer_circuit, inplace=True)

            # === Trainable Layer ===
            unitary_layer_circuit = QuantumCircuit(num_qubits)
            for i in range(num_qubits):
                circuit_unitary, unitary_parameters = create_unitary(
                    i, layer, rot_per_unitary
                )
                # Add the 1-qubit unitary to the correct qubit (i)
                unitary_layer_circuit.compose(circuit_unitary, [i], inplace=True)
                trainable_symbols.extend(unitary_parameters)

            circuit.compose(unitary_layer_circuit, inplace=True)
            circuit.compose(create_entanglement(num_qubits), inplace=True)

    return input_symbols, trainable_symbols, circuit