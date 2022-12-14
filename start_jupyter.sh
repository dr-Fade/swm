# start the jupyter server in the src dir
ROOT_DIR=$(dirname "$(readlink -f "$0")")
jupyter lab --notebook-dir=$ROOT_DIR/src --preferred-dir=$ROOT_DIR/src