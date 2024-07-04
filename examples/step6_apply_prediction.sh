## After prediction, compounds can be aligned back to the probability map
## to predict prefered poses.
PROJ_HOME=FeatureDock
SCRIPT_HOME=${PROJ_HOME}/src
PRED_DIR=${PROJ_HOME}/examples
name=1b38
PRED_COMP_DIR=${PRED_DIR}/compounds

SEED=42
NCONFS=20
CONF_RMSD=1.0
NSAMPLES=500
VS_RESULT=${PRED_DIR}/${name}_vs_results_${NCONFS}_${NSAMPLES}_seed${SEED}
mkdir -p ${VS_RESULT}
CLS_RMSD=1.0
CLS_STATS=avg

comp_name=CHEMBL402158
################################
#    PREDICT AND SCORE POSES   #
################################
python ${SCRIPT_HOME}/application/virtual_screen.py \
    --voxelfile=${PRED_DIR}/aligned_merged_pocket/${name}.voxels.pkl \
    --seed=${SEED} \
    --probfile=${PRED_DIR}/aligned_merged_pocket_transformer_20/ensemble_average/${name}.property.ensemble_predictions.pkl \
    --outdir=${VS_RESULT}/${comp_name} \
    --embedding \
    --sdffile=${PRED_COMP_DIR}/${comp_name}.sdf \
    --nconfs=${NCONFS} \
    --conf_rmsd=${CONF_RMSD} \
    --nsamples=${NSAMPLES} \
    --threads=10

################################
#    CLUSTER PREDICTED POSES   #
################################
resultfile=${VS_RESULT}/${comp_name}/${comp_name}.${NCONFS}confs.${NSAMPLES}attempts.results.pkl
grpfile=${VS_RESULT}/${comp_name}/${comp_name}.${NCONFS}confs.${NSAMPLES}attempts.dbscan_results_${CLS_RMSD}_${CLS_STATS}.pkl

python ${SCRIPT_HOME}/application/dbscan_cluster.py \
    --seed=${SEED} \
    --resultfile=${resultfile} \
    --outfile=${grpfile} \
    --cls_rmsd=${CLS_RMSD} \
    --cls_stats=${CLS_STATS}

#############################
#    PLOT PREDICTED POSES   #
#############################
python ${SCRIPT_HOME}/application/plot_predicted_poses.py \
    --grp_file=${grpfile} \
    --sdffile=${PRED_COMP_DIR}/${comp_name}.sdf \
    --topK=5 \
    --outdir=${VS_RESULT}/${comp_name}