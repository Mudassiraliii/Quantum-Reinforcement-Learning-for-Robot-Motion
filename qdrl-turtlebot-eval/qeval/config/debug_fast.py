import tensorflow as tf

CONFIG = {
    "ENV": "CartPole-v1",
    "TYPE": "quantum",

    "N_ITERATIONS": 500,
    "COLLECT_STEPS": 1,
    "MAX_STEPS": 200,

    # Drastically reduced for speed
    "INITIAL_COLLECT": 64,
    "EVAL_RUNS": 1,
    "EVAL_EVERY": 10,
    "EVAL_THRESHOLD": 475.0,

    "BATCH_SIZE": 64,
    "RB_CAPACITY": 10000,

    "EPSILON_START": 1.0,
    "EPSILON_MIN": 0.05,
    "EPSILON_DECAY": 0.95,
    "DECAY_EVERY": 50,

    "TARGET_UPDATE": 10,
    "GAMMA": 0.99,

    "INPUT_LRATE": 0.01,
    "CIRCUIT_LRATE": 0.001,
    "OUTPUT_LRATE": 0.01,

    "QUANTUM_MODEL_CONFIG": {
        "n_qubits": 4,
        # Reduced from 10 to 1 for speed testing
        "layers": 1,
        "input_style": 1,
        "rot_per_unitary": 3,
        "data_reupload": True,
        "trainable_input": True,
        "zero_layer": True,
        "rescale": False,
        "activition": tf.math.atan,
        "n_states": 4,
        "n_actions": 2
    }
}