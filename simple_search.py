import itertools
import subprocess

# Hyperparams 
learning_rates = [0.01, 0.001]
epochs = [20, 40]
batch_sizes = [64, 128]

search_space = itertools.product(learning_rates, epochs, batch_sizes)

for job, (learning_rate, num_epochs, batch_size_per_replica) in enumerate(search_space):
    subprocess.run(["dvc", "exp", "run", "--queue",
        "--set-param", f"volcano.job_id={job}",
        "--set-param", f"train.batch_size_per_replica={batch_size_per_replica}",
        "--set-param", f"train.epochs={num_epochs}",
        "--set-param", f"train.learning_rate={learning_rate}",
    ])