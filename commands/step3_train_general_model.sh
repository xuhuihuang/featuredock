################################
# WO LR DECAY AND WEIGHT DECAY #
################################
PROJ_HOME=FeatureDock
DATA_DIR=${PROJ_HOME}/data
SCRIPT_HOME=${PROJ_HOME}/src
VOXEL_DIR=${DATA_DIR}/voxels
# whole dataset
SQID=90
DATACLAN=${DATA_DIR}/ClanGraph_${SQID}_df.pkl
MODELREPO=${PROJ_HOME}/results
mkdir -p ${MODELREPO}

#############
#  TRAINING #
#############
## Transformer
SEED=0
TASK=HeavyAtomsite
MODELTYPE=transformer
NBLOCKS=1
LR=0.01
EPOCHS=10
MODELNAME=${TASK}_${MODELTYPE}_${NBLOCKS}_seed${SEED}
mkdir -p ${MODELREPO}/${MODELNAME}


python -u ${SCRIPT_HOME}/models/train_main.py \
  --modeltype=${MODELTYPE} \
  --seed=${SEED} \
  --task=${TASK} \
  --modelname=${MODELNAME} \
  --n_blocks=${NBLOCKS} \
  --optimizer=AdamW \
  --scheduler=plateau \
  --lr=${LR} \
  --steps=${EPOCHS} \
  --n_structs=5 \
  --n_resamples=1000 \
  --weight_decay=0 \
  --save_every=10 \
  --earlystop \
  --patience=${EPOCHS} \
  --graphclan=${DATACLAN} \
  --datafolder=${VOXEL_DIR} \
  --outfolder=${MODELREPO}/${MODELNAME} \
  --verbose \
  --use_gpu > ${MODELREPO}/${MODELNAME}/${MODELNAME}.LOG
