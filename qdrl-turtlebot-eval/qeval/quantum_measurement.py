from qiskit.quantum_info import SparsePauliOp

def create_measurement(num_qubits=3, env='qturtle'):
    """
    Creates a list of measurement operators (observables) using Qiskit.

    Args:
        num_qubits: The number of qubits in the circuit.
        env: The name of the environment, which determines the measurement strategy.

    Returns:
        A list of SparsePauliOp objects representing the observables.
    """
    measurement = []

    if env == 'FrozenLake-v1' and num_qubits == 4:
        # Z measurement on each of the 4 qubits
        measurement.append(SparsePauliOp('IIIZ'))
        measurement.append(SparsePauliOp('IIZI'))
        measurement.append(SparsePauliOp('IZII'))
        measurement.append(SparsePauliOp('ZIII'))

    elif env == 'CartPole-v1' and num_qubits == 4:
        # ZZ measurement on qubit pairs (0,1) and (2,3)
        measurement.append(SparsePauliOp('IIZZ'))
        measurement.append(SparsePauliOp('ZZII'))

    elif num_qubits == 3 or num_qubits == 12:
        # Z measurement on the first 3 qubits
        
        # Observable for Z on qubit 0
        op_0 = ['I'] * num_qubits
        op_0[0] = 'Z'
        measurement.append(SparsePauliOp("".join(reversed(op_0))))

        # Observable for Z on qubit 1
        op_1 = ['I'] * num_qubits
        op_1[1] = 'Z'
        measurement.append(SparsePauliOp("".join(reversed(op_1))))

        # Observable for Z on qubit 2
        op_2 = ['I'] * num_qubits
        op_2[2] = 'Z'
        measurement.append(SparsePauliOp("".join(reversed(op_2))))

    else:
        raise ValueError(f"Measurement not defined for env {env} and num_qubits {num_qubits}.")

    return measurement