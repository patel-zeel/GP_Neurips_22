from time import time
import os
import sys

method = sys.argv[1]
gpu_id = sys.argv[2]

# create a new log file in linux
os.system(f"touch {method}.log")

init = time()
for latent_seed, data_seed in zip(range(200, 220), range(300, 320)):
    os.system(
        f"python -u ablations.py {latent_seed} {data_seed} {method} {gpu_id} >> {method}.log"
    )

print(f"Total time: {(time() - init)/60} minutes")
