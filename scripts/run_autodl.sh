# !/usr/bin/env bash
# set -euo pipefail

# export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# MODE="${MODE:-online}"
# STEPS="${STEPS:-100000}"
# SEEDS="${SEEDS:-0 1 2 3}"
# TASKS=(
#   "ShadowHandBottleCap"
#   "ShadowHandDoorOpenInward"
#   "ShadowHandDoorOpenOutward"
#   "ShadowHandPen"
# )

# for seed in ${SEEDS}; do
#   for task in "${TASKS[@]}"; do
#     echo "Running DIMA on bidexhands/${task} with seed ${seed} for ${STEPS} steps"
#     python3 train.py \
#       --env bidexhands \
#       --env_name "${task}" \
#       --policy_class gaussian \
#       --seed "${seed}" \
#       --steps "${STEPS}" \
#       --mode "${MODE}"
#   done
# done


# CUDA_VISIBLE_DEVICES=1 /usr/bin/python3 train.py --env mamujoco --env_name HalfCheetah-v2 --seed 0 --agent_conf 2x3 --steps 100000 --mode offline --use_mamba --use_harmony --mamba_d_state 16 --mamba_d_model 256 --mamba_n_layers 3 --mamba_expand 2 --mamba_d_conv 3 --rec_emb_lr_mult 2 --backend jax ##train each 200 #10k step warmup #1000 ac first update # ac epoch 10

declare -a scenarios=(
  "Ant-v2 2x4"
  "Ant-v2 4x2"
  "HalfCheetah-v2 2x3"
  "HalfCheetah-v2 3x2"
  "HalfCheetah-v2 6x1"
  "Walker2d-v2 2x3"
  "Walker2d-v2 3x2"
)
TASKS=(
  "ShadowHandBottleCap"
  "ShadowHandDoorOpenInward"
  "ShadowHandDoorOpenOutward"
  "ShadowHandPen"
)

is_cuda_python_running() {
  local cuda_id="$1"
  local pid

  for pid in $(pgrep -x "python3|pt_main_thread"); do
    if tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=${cuda_id}"; then
      if ps -p "$pid" -o args= 2>/dev/null | grep -q "train.py"; then
        return 0
      fi
    fi
  done

  return 1
}

submit_job() {
  local -a cmd=("$@")
  local found_available

  while true; do
    found_available=0
    for cuda_id in 0; do
      if ! is_cuda_python_running "$cuda_id"; then
        echo "cuda:${cuda_id} is availible"
        CUDA_VISIBLE_DEVICES=${cuda_id} nohup "${cmd[@]}" > "cuda${cuda_id}.log" 2>&1 &
        found_available=1
        sleep 20
        break
      fi
    done

    if [ "$found_available" -eq 1 ]; then
      break
    fi

    sleep 5
  done
}

CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 train.py --env mamujoco --env_name HalfCheetah-v2 --agent_conf 2x3 --seed 0 --steps 10000 --mode offline 
submit_job /usr/bin/python3 train.py --env mamujoco --env_name HalfCheetah-v2 --agent_conf 2x3 --seed 0 --steps 10000 --mode offline  
# submit_job /usr/bin/python3 train.py --env mamujoco --env_name HalfCheetah-v2 --agent_conf 3x2 --seed 0 --steps 100000 --mode online   
# submit_job /usr/bin/python3 train.py --env mamujoco --env_name HalfCheetah-v2 --agent_conf 6x1 --seed 0 --steps 100000 --mode online   
# submit_job /usr/bin/python3 train.py --env mamujoco --env_name Walker2d-v2 --agent_conf 2x3 --seed 0 --steps 100000 --mode online   
# submit_job /usr/bin/python3 train.py --env mamujoco --env_name Walker2d-v2 --agent_conf 3x2 --seed 0 --steps 100000 --mode online  
# submit_job /usr/bin/python3 train.py --env mamujoco --env_name Ant-v2 --agent_conf 2x4 --seed 0 --steps 100000 --mode online 
# submit_job /usr/bin/python3 train.py --env mamujoco --env_name Ant-v2 --agent_conf 4x2 --seed 0 --steps 100000 --mode online  

# /root/miniconda3/envs/dima/bin/python
# /usr/bin/python3 train.py --env mamujoco --env_name HalfCheetah-v2 --agent_conf 2x3 --seed 0 --steps 100000 --mode offline  

# CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/dima/bin/python train.py --env mamujoco --env_name Ant-v2 --agent_conf 4x2 --seed 0 --steps 100000 --mode online 
# CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/dima/bin/python train.py --env mamujoco --env_name HalfCheetah-v2 --agent_conf 2x3 --seed 3 --steps 100000 --mode online  
# CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/dima/bin/python train.py --env mamujoco --env_name HalfCheetah-v2 --agent_conf 3x2 --seed 3 --steps 100000 --mode online   
# CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/dima/bin/python train.py --env mamujoco --env_name HalfCheetah-v2 --agent_conf 6x1 --seed 1 --steps 100000 --mode online   
# CUDA_VISIBLE_DEVICES=1 /root/miniconda3/envs/dima/bin/python train.py --env mamujoco --env_name Walker2d-v2 --agent_conf 3x2 --seed 1 --steps 100000 --mode online  
# CUDA_VISIBLE_DEVICES=1 /root/miniconda3/envs/dima/bin/python train.py --env mamujoco --env_name Ant-v2 --agent_conf 4x2 --seed 1 --steps 100000 --mode online  


# CUDA_VISIBLE_DEVICES=1 /root/miniconda3/envs/dima/bin/python train.py --env mamujoco --env_name HalfCheetah-v2 --agent_conf 6x1 --seed 3 --steps 100000 --mode online  
# CUDA_VISIBLE_DEVICES=1 /root/miniconda3/envs/dima/bin/python train.py --env mamujoco --env_name HalfCheetah-v2 --agent_conf 2x3 --seed 1 --steps 100000 --mode online  
# CUDA_VISIBLE_DEVICES=2 /root/miniconda3/envs/dima/bin/python train.py --env mamujoco --env_name HalfCheetah-v2 --agent_conf 3x2 --seed 1 --steps 100000 --mode online   
# CUDA_VISIBLE_DEVICES=2 /root/miniconda3/envs/dima/bin/python train.py --env mamujoco --env_name Walker2d-v2 --agent_conf 2x3 --seed 1 --steps 100000 --mode online   
# CUDA_VISIBLE_DEVICES=2 /root/miniconda3/envs/dima/bin/python train.py --env mamujoco --env_name Ant-v2 --agent_conf 2x4 --seed 1 --steps 100000 --mode online 
# CUDA_VISIBLE_DEVICES=2 /root/miniconda3/envs/dima/bin/python train.py --env mamujoco --env_name HalfCheetah-v2 --agent_conf 2x3 --seed 2 --steps 100000 --mode online  


# CUDA_VISIBLE_DEVICES=2 /root/miniconda3/envs/dima/bin/python train.py --env mamujoco --env_name HalfCheetah-v2 --agent_conf 3x2 --seed 2 --steps 100000 --mode online   
# CUDA_VISIBLE_DEVICES=3 /root/miniconda3/envs/dima/bin/python train.py --env mamujoco --env_name HalfCheetah-v2 --agent_conf 6x1 --seed 2 --steps 100000 --mode online   
# CUDA_VISIBLE_DEVICES=3 /root/miniconda3/envs/dima/bin/python train.py --env mamujoco --env_name Walker2d-v2 --agent_conf 2x3 --seed 2 --steps 100000 --mode online   
# CUDA_VISIBLE_DEVICES=3 /root/miniconda3/envs/dima/bin/python train.py --env mamujoco --env_name Walker2d-v2 --agent_conf 3x2 --seed 2 --steps 100000 --mode online  
# CUDA_VISIBLE_DEVICES=3 /root/miniconda3/envs/dima/bin/python train.py --env mamujoco --env_name Ant-v2 --agent_conf 2x4 --seed 2 --steps 100000 --mode online 
# CUDA_VISIBLE_DEVICES=3 /root/miniconda3/envs/dima/bin/python train.py --env mamujoco --env_name Ant-v2 --agent_conf 4x2 --seed 2 --steps 100000 --mode online  