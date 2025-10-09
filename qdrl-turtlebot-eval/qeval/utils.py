import os
import pickle
from datetime import datetime

import tensorflow as tf  # type: ignore


def set_gpu_memory_growing():
    """
    Safely configures GPU memory growth. If no GPU is found,
    it prints a message and allows the program to continue on the CPU.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        # If any GPUs are found, try to configure them
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found and configured {len(gpus)} GPU(s).")
        except RuntimeError as e:
            # This error can happen if memory growth is already set
            print(f"Error configuring GPU: {e}")
    else:
        # If the list of GPUs is empty, do nothing and proceed on CPU
        print("No GPU found. The program will continue on the CPU.")


def create_results_dir(dirname):
    try:
        os.makedirs(dirname, exist_ok=True)
    except OSError as err:
        raise RuntimeError(f"Could not create target directories: {err}")
    else:
        print(f"Directory {dirname} created.")


def save_results(n, args, config, rewards):
    time = datetime.now().strftime("%d-%m-%Y_%H-%M")

    res = {
        "time": time,
        "n": n,
        "config": config.__dict__,
        "rewards": rewards,
    }

    filename = f"{args.dir}/{args.config}_run_{n:02d}_{time}.pickle"

    with open(filename, "wb") as f:
        pickle.dump(res, f)

    return filename


def save_run_statistics(n, args, steps, final_reward):
    filename = f"{args.dir}/stats.txt"

    with open(filename, "a") as f:
        f.write(f"{n}, {steps}, {final_reward}\n")
