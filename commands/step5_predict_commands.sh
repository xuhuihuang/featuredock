PROJ_HOME=FeatureDock
DATA_DIR=${PROJ_HOME}/data
SCRIPT_HOME=${PROJ_HOME}/src
VOXEL_DIR=${DATA_DIR}/voxels
MODELREPO=${PROJ_HOME}/results
FEATURE_PROGRAM=${PROJ_HOME}/src/utils/feature-3.1.0

PRED_DIR=${PROJ_HOME}/examples
name=1b38

###############################
#  PREPARE QUERY GRIDPOINTS   #
###############################
python ${SCRIPT_HOME}/application/prepare_query_box.py \
    --name=${name} \
    --xmin=-7.62 \
    --xmax=10.38 \
    --ymin=18.15 \
    --ymax=34.14 \
    --zmin=1.40 \
    --zmax=17.40 \
    --outdir=${PRED_DIR} \
    --intermediate \
    --spacing=1.0


echo "Grid points can be further filtered based on the distance to protein atoms, "
echo "or the distance to existing ligands in cocrystal structures, or both."
echo "An example of filtered grid points: examples/aligned_merged_pocket/1b38.voxels.pkl"


#################################
#  FEATURIZE QUERY GRIDPOINTS   #
#################################
dssp -i ${PRED_DIR}/${name}.pdb -o ${PRED_DIR}/${name}.dssp
python ${SCRIPT_HOME}/curate_dataset/featurize.py \
    --pdbid=${name} \
    --voxelfile=${PRED_DIR}/${name}_gridpoints.voxels.pkl \
    --voxeldir=${PRED_DIR} \
    --tempdir=${PRED_DIR} \
    --searchdir=${PRED_DIR} \
    --featurize=${FEATURE_PROGRAM} \
    --numshell=6 \
    --width=1.25 \
    --overwrite


###############################
#    PREDICT HEAVYATOMSITES   #
###############################
SEED=0
TASK=HeavyAtomsite
MODELTYPE=transformer
NBLOCKS=20
MODELNAME=${TASK}_${MODELTYPE}_${NBLOCKS}_seed${SEED}
final_params=$(echo ${MODELREPO}/${MODELNAME}/*_final_params.torch)
## output probability file
mkdir -p ${PRED_DIR}/${name}_predictions

python src/application/predict_main.py \
    --configfile=${MODELREPO}/${MODELNAME}/${MODELNAME}_config.torch \
    --paramfile=${MODELREPO}/${MODELNAME}/${MODELNAME}_best_checkpoint_params.torch \
    --datafile=${PRED_DIR}/${name}.property.pvar \
    --outfile=${PRED_DIR}/${name}_predictions/${MODELNAME}_${name}.predictions.pkl \
    --batchsize=10000 \
    --use_gpu

###############################
#     ENSEMBL PREDICTIONS     #
###############################
python ${SCRIPT_HOME}/application/ensemble_average.py \
    --indir=${PRED_DIR}/${name}_predictions \
    --outfile=${PRED_DIR}/${name}.ensemble.predictions.pkl


###############################
#     PYMOL VISUALIZATIOM     #
###############################
# plot grid points with probabilities above 0.7, 0.8, 0.9, 0.95
# the output xyz files can be visualized in PyMol
python ${SCRIPT_HOME}/application/plot_prediction.py \
    --voxelfile=${PRED_DIR}/${name}_gridpoints.voxels.pkl \
    --probfile=${PRED_DIR}/${name}.ensemble.predictions.pkl \
    --outdir=${PRED_DIR}/${name}_xyz \
    --cutoffs 0.8 0.9 0.95

echo "plot colored probability map in PyMol"
echo "Run src/application/plot_probability_map.py in PyMol"
echo "Run plot_probability(probfile, voxelfile, cutoff=0.8, colormap='Blues', relative=True, is_rank=False, plot_every=3) in the PyMol command line"
