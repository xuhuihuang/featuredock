
PROJ_HOME=FeatureDock
DATA_DIR=${PROJ_HOME}/data
SCRIPT_HOME=${PROJ_HOME}/src
VOXEL_DIR=${DATA_DIR}/voxels
MODELREPO=${PROJ_HOME}/results

###############################
#   EVALUATE HEAVYATOM SITES  #
#      WHOLE DATASET          #
###############################
SEED=0
TASK=HeavyAtomsite
MODELTYPE=transformer
NBLOCKS=20
MODELNAME=${TASK}_${MODELTYPE}_${NBLOCKS}_seed${SEED}

final_params=$(echo ${MODELREPO}/${MODELNAME}/*_final_params.torch)
python src/models/evaluate_main.py \
    --seed=${SEED} \
    --configfile=${MODELREPO}/${MODELNAME}/${MODELNAME}_config.torch \
    --paramfile=${MODELREPO}/${MODELNAME}/${MODELNAME}_best_checkpoint_params.torch \
    --datafolder=${VOXEL_DIR} \
    --use_gpu > ${MODELREPO}/${MODELNAME}/${MODELNAME}_evaluation.LOG
