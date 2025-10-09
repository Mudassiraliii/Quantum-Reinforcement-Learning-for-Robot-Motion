import qiskit
from qiskit.circuit import QuantumCircuit, Parameter

def get_new_param(symbol_name: str, qubit: int, position: int, layer: int = None):
    """
    Returns a new Qiskit learnable parameter.
    """
    if layer is not None:
        name = f"{symbol_name}_{qubit}_{layer}_{position}"
    else:
        name = f"{symbol_name}_{qubit}_{position}"
    return Parameter(name)

def create_input(qubit: int, n_qubit: int, symbol_name: str, layer: int = None, input_style: int = 1):
    """
    Creates the input encoding part of the circuit for a single qubit.
    """
    circuit = QuantumCircuit(n_qubit + 1) # Ensure circuit is large enough
    input_parameters = []

    if input_style == 1:
        input_parameter = get_new_param(symbol_name, n_qubit, 0, layer)
        circuit.rx(input_parameter, qubit)
        input_parameters.append(input_parameter)

    elif input_style == 2:
        input_parameter_0 = get_new_param(symbol_name, n_qubit, 0, layer)
        input_parameter_1 = get_new_param(symbol_name, n_qubit, 1, layer)

        circuit.rx(input_parameter_0, qubit)
        circuit.ry(input_parameter_1, qubit)

        input_parameters.extend([input_parameter_0, input_parameter_1])

    elif input_style == 3:
        input_parameter_0 = get_new_param(symbol_name, n_qubit, 0, layer)
        input_parameter_1 = get_new_param(symbol_name, n_qubit, 1, layer)
        input_parameter_2 = get_new_param(symbol_name, n_qubit, 2, layer)

        circuit.rx(input_parameter_0, qubit)
        circuit.ry(input_parameter_1, qubit)
        circuit.rx(input_parameter_2, qubit)

        input_parameters.extend([input_parameter_0, input_parameter_1, input_parameter_2])
    else:
        raise ValueError(f"input_style not implemented: {input_style}")

    return circuit, tuple(input_parameters)

def create_entanglement(num_qubits: int):
    """
    Creates an entanglement layer.
    """
    circuit = QuantumCircuit(num_qubits)
    if num_qubits > 1:
        for i in range(num_qubits - 1):
            circuit.cz(i, i + 1)

    if num_qubits > 2:
        circuit.cz(num_qubits - 1, 0)

    return circuit

def create_unitary(qubit: int, n_qubit: int, layer: int, rot_per_unitary: int = 3, circuit_type: str = "skolik"):
    """
    Creates a unitary block for a single qubit.
    """
    circuit = QuantumCircuit(n_qubit + 1) # Ensure circuit is large enough
    body_parameters = []
    symbol_name = "train"

    if rot_per_unitary == 2:
        body_parameter_0 = get_new_param(symbol_name, n_qubit, 0, layer)
        body_parameter_1 = get_new_param(symbol_name, n_qubit, 1, layer)

        circuit.ry(body_parameter_0, qubit)
        circuit.rz(body_parameter_1, qubit)

        body_parameters.extend([body_parameter_0, body_parameter_1])

    elif rot_per_unitary == 3 and circuit_type == "skolik":
        body_parameter_0 = get_new_param(symbol_name, n_qubit, 0, layer)
        body_parameter_1 = get_new_param(symbol_name, n_qubit, 1, layer)
        body_parameter_2 = get_new_param(symbol_name, n_qubit, 2, layer)

        circuit.rx(body_parameter_0, qubit)
        circuit.ry(body_parameter_1, qubit)
        circuit.rz(body_parameter_2, qubit)

        body_parameters.extend([body_parameter_0, body_parameter_1, body_parameter_2])

    elif rot_per_unitary == 3 and circuit_type == "han":
        body_parameter_0 = get_new_param(symbol_name, n_qubit, 0, layer)
        body_parameter_1 = get_new_param(symbol_name, n_qubit, 1, layer)
        body_parameter_2 = get_new_param(symbol_name, n_qubit, 2, layer)

        circuit.rx(body_parameter_0, qubit)
        circuit.ry(body_parameter_1, qubit)
        circuit.rx(body_parameter_2, qubit)

        body_parameters.extend([body_parameter_0, body_parameter_1, body_parameter_2])
    else:
        raise ValueError("rot_per_unitary must be 2 or 3")

    return circuit, tuple(body_parameters)

def create_reupload_circuit(
    num_qubits: int,
    layers: int,
    rot_per_unitary: int = 3,
    input_style: int = 3,
    data_reupload: bool = True,
    trainable_input: bool = True,
    zero_layer: bool = True,
):
    """
    Builds the complete PQC by composing input, unitary, and entanglement layers.
    """
    circuit = QuantumCircuit(num_qubits, name="PQC")
    
    input_symbols_name = "in_train" if trainable_input else "in"
    
    input_symbols = ()
    trainable_symbols = ()

    for layer in range(layers + 1):
        if layer == 0 and zero_layer:
            for i in range(num_qubits):
                circuit_unitary, unitary_parameters = create_unitary(i, i, layer, rot_per_unitary)
                circuit.compose(circuit_unitary, inplace=True)
                trainable_symbols += unitary_parameters
            circuit.compose(create_entanglement(num_qubits), inplace=True)

        elif layer > 0:
            if layer == 1 or data_reupload:
                for i in range(num_qubits):
                    input_circuit, input_parameters = create_input(i, i, input_symbols_name, layer, input_style)
                    circuit.compose(input_circuit, inplace=True)
                    input_symbols += input_parameters

            for i in range(num_qubits):
                circuit_unitary, unitary_parameters = create_unitary(i, i, layer, rot_per_unitary)
                circuit.compose(circuit_unitary, inplace=True)
                trainable_symbols += unitary_parameters
            
            circuit.compose(create_entanglement(num_qubits), inplace=True)

    return input_symbols, trainable_symbols, circuit