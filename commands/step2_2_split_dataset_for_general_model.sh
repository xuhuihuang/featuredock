PROJ_HOME=FeatureDock
DATA_DIR=${PROJ_HOME}/data
SCRIPT_HOME=${PROJ_HOME}/src
VOXEL_DIR=${DATA_DIR}/voxels
LABELEDLIST=${DATA_DIR}/labeled_pdblist.txt
SQID=90
SEED=42
DATACLAN=${DATA_DIR}/ClanGraph_${SQID}.pkl
CLANFILE=${DATA_DIR}/ClanGraph_${SQID}_df.pkl

########################
#  CLUSTER STRUCTURES  #
########################
python ${SCRIPT_HOME}/curate_dataset/cluster_structures.py \
    --sqid=${SQID} \
    --outdir=${DATA_DIR} \
    --pdbids=${LABELDLIST} \
    --overwrite \
    --draw

