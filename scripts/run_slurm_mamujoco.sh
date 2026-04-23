# set -x
# PARTITION=${PARTITION:-"optimal"}
# GPUS_PER_NODE=${GPUS_PER_NODE:-1}
# export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# seeds=(
#     1 2 3 4
# )

# # HalfCheetah-v2
# env="mamujoco"
# map_name="Ant-v2"   # Ant-v2
# agent_conf="4x2"            # 4x2
# steps=1000000

# for seed in "${seeds[@]}"; do
#     date_dir=$(date "+%Y-%m-%d")
#     cur_date=$(date "+%H-%M-%S")
#     OUTPUT_DIR=training-runs/$date_dir

#     # log_name=$(echo "$map_name" | awk -F'_' '{print $(NF-2) "-" $(NF-1) "-" $NF}')
#     log_name="$map_name-$agent_conf-seed_$seed"
#     echo $log_name
#     note=

#     mkdir -p $OUTPUT_DIR

#     sbatch -p ${PARTITION} \
#     -J ${log_name}${note:+-$note} \
#     -N 1 \
#     -n 6 \
#     -o ${OUTPUT_DIR}/${cur_date}-${log_name}${note:+-$note}-%j.out \
#     --gres=gpu:${GPUS_PER_NODE} \
#     --wrap="python train.py \
#             --n_workers 1 \
#             --env $env \
#             --env_name $map_name \
#             --policy_class gaussian \
#             --seed $seed \
#             --agent_conf $agent_conf \
#             --steps $steps \
#             --mode offline \
#             --temperature 1.0 \
#             --sample_temp 20 \
#             --state_decoder_type 1 \
#             --ce_for_cont --use_tensorboard"
# done

# submit_job /usr/bin/python3 train.py --env mamujoco --env_name HalfCheetah-v2 --agent_conf 2x3 --seed 0 --steps 100000 --mode online 
# submit_job /usr/bin/python3 train.py --env mamujoco --env_name HalfCheetah-v2 --agent_conf 3x2 --seed 0 --steps 100000 --mode online 
# submit_job /usr/bin/python3 train.py --env mamujoco --env_name HalfCheetah-v2 --agent_conf 6x1 --seed 0 --steps 100000 --mode online  
# submit_job /usr/bin/python3 train.py --env mamujoco --env_name Walker2d-v2 --agent_conf 2x3 --seed 0 --steps 100000 --mode online 
# submit_job /usr/bin/python3 train.py --env mamujoco --env_name Walker2d-v2 --agent_conf 3x2 --seed 0 --steps 100000 --mode online
# submit_job /usr/bin/python3 train.py --env mamujoco --env_name Ant-v2 --agent_conf 2x4 --seed 0 --steps 100000 --mode online 
# submit_job /usr/bin/python3 train.py --env mamujoco --env_name Ant-v2 --agent_conf 4x2 --seed 0 --steps 100000 --mode offline  

#  /root/miniconda3/envs/py38/bin/python3 train.py --env bidexhands --env_name ShadowHandBottleCap --seed 0 --steps 100000 --mode online --sim_device cpu --pipeline cpu
#  /root/miniconda3/envs/py38/bin/python3 train.py --env bidexhands --env_name ShadowHandDoorOpenInward --seed 0 --steps 100000 --mode online
#  /root/miniconda3/envs/py38/bin/python3 train.py --env bidexhands --env_name ShadowHandDoorOpenOutward --seed 0 --steps 100000 --mode online
#  /root/miniconda3/envs/py38/bin/python3 train.py --env bidexhands --env_name ShadowHandPen --seed 0 --steps 100000 --mode online