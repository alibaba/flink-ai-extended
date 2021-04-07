#!/usr/bin/env bash
ROOT=$(cd "$(dirname "$0")";pwd)
echo $ROOT
export PYTHONPATH=${ROOT}

python ${ROOT}/census_distribute.py $1 $2