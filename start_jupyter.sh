# start the jupyter server in the src dir
ROOT_DIR=$(dirname "$(readlink -f "$0")")
cd $ROOT_DIR
JULIA_CUDA_SOFT_MEMORY_LIMIT=50% jupyter lab  --ip="0.0.0.0" --notebook-dir=$(pwd)/src --preferred-dir=$(pwd)/src
