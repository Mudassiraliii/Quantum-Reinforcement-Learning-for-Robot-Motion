from qiskit.quantum_info import SparsePauliOp


def create_measurement(num_qubits=3, env='qturtle'):
    """
    Creates the measurement observables as a list of Qiskit SparsePauliOp.
    """
    measurement = []

    if env == 'FrozenLake-v1' and num_qubits == 4:
        # Replaces:
        # cirq.Z(qubits[0]) -> "IIIZ" (Z on qubit 0, I on 1, 2, 3)
        # cirq.Z(qubits[1]) -> "IIZI"
        # cirq.Z(qubits[2]) -> "IZII"
        # cirq.Z(qubits[3]) -> "ZIII"
        # Note: Qiskit's string order is read from right-to-left (qubit 0 is the rightmost char).
        measurement.append(SparsePauliOp("IIIZ"))
        measurement.append(SparsePauliOp("IIZI"))
        measurement.append(SparsePauliOp("IZII"))
        measurement.append(SparsePauliOp("ZIII"))

    elif env == 'CartPole-v1' and num_qubits == 4:
        # Replaces:
        # cirq.Z(qubits[0]) * cirq.Z(qubits[1]) -> "IIZZ"
        # cirq.Z(qubits[2]) * cirq.Z(qubits[3]) -> "ZZII"
        measurement.append(SparsePauliOp("IIZZ"))
        measurement.append(SparsePauliOp("ZZII"))

    elif num_qubits == 3:
        # Replaces cirq.Z(qubits[0]), Z(qubits[1]), Z(qubits[2])
        measurement.append(SparsePauliOp("IIZ"))
        measurement.append(SparsePauliOp("IZI"))
        measurement.append(SparsePauliOp("ZII"))
        
    elif num_qubits == 12:
        # This is a bit ambiguous, but assuming Z on first 3 qubits
        measurement.append(SparsePauliOp("I" * 11 + "Z")) # Z on qubit 0
        measurement.append(SparsePauliOp("I" * 10 + "ZI")) # Z on qubit 1
        measurement.append(SparsePauliOp("I" * 9 + "ZII")) # Z on qubit 2

    else:
        raise ValueError(
            f"Measurement not defined for env {env} and num_qubits {num_qubits}."
        )

    return measurement