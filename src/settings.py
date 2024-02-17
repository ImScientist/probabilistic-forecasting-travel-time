from environs import Env

env = Env()

DATA_DIR = env.str("DATA_DIR", "data")
TFBOARD_DIR = env.str("TFBOARD_DIR", "tfboard")
ARTIFACTS_DIR = env.str("ARTIFACTS_DIR", "artifacts")

# maximum memory in GB that can be allocated by tensorflow
GPU_MEMORY_LIMIT = env.int("GPU_MEMORY_LIMIT", 16)
