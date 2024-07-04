# This project starts from cleaned protein file (in pdb format) and cleaned ligand file (in sdf format),
# so that the bond types of ligands needn't additional mapping.
# The only thing needed to do before create interaction map is to remove waters.


###########################
#  ENVIRONMENT VARIABLES  #
###########################
# absolute path to project directory
PROJ_HOME=FeatureDock
DATA_DIR=${PROJ_HOME}/data

## dependent paths
SCRIPT_HOME=${PROJ_HOME}/src
PDBBIND_DIR=${DATA_DIR}/refined-set
APO_DIR=${DATA_DIR}/apo
HET_DIR=${DATA_DIR}/het
VOXEL_DIR=${DATA_DIR}/voxels
BACKUP_DIR=${DATA_DIR}/voxel_details
PDBLIST=${DATA_DIR}/pdblist.txt
LABELDLOG=${DATA_DIR}/labeled.LOG
LABELDLIST=${DATA_DIR}/labeled_pdblist.txt
FEATURE_PROGRAM=${PROJ_HOME}/src/utils/feature-3.1.0
FF_DIR=${DATA_DIR}/ff
mkdir -p ${APO_DIR} ${HET_DIR}
mkdir -p ${VOXEL_DIR} ${FF_DIR} ${BACKUP_DIR}

###########################
#   PREPARE STRUCTURES    #
###########################
## Parallel run
cat ${PDBLIST} | parallel -j 8 \
    python ${SCRIPT_HOME}/curate_dataset/prepare_structure.py \
        --pdbid={} \
        --protfile=${PDBBIND_DIR}/{}/{}_protein.pdb \
        --ligfile=${PDBBIND_DIR}/{}/{}_ligand.sdf \
        --apodir=${APO_DIR} \
        --hetdir=${HET_DIR} \
        --dssp=dssp \
        --rm_all_het


##################################
#   CREATE VOXELS AND LANDMARKS  #
##################################
## Parallel run
# voxels will be overwritten when --overwrite is flagged,
# but landmarks will be created no matter what
cat ${PDBLIST} | parallel -j 8 \
    python ${SCRIPT_HOME}/curate_dataset/create_voxels_and_landmarks.py \
        --pdbid={} \
        --apofile=${APO_DIR}/{}.pdb \
        --hetfile=${HET_DIR}/{}_ligand.sdf \
        --outdir=${BACKUP_DIR} \
        --pocket_cutoff=6.0 \
        --spacing=1.0 \
        --trim \
        --trim_min=1.0 \
        --trim_max=6.0 \
        --abs_include \
        --heavyatom \
        --intermediate \
        --overwrite \
> ${LABELDLOG}

# move *.voxels.pkl to voxel folder
mv ${BACKUP_DIR}/*.voxels.pkl ${VOXEL_DIR}/
mv ${BACKUP_DIR}/*.landmarks.pkl ${VOXEL_DIR}/
cat ${LABELDLOG} | grep "Succesfully" | cut -d':' -f2 | sed -r "s/\s+//g" > ${LABELDLIST}

######################
#   CREATE FEATURES  #
######################
cat ${LABELDLIST} | parallel -j 8 \
    python ${SCRIPT_HOME}/curate_dataset/featurize.py \
        --pdbid={} \
        --voxelfile=${VOXEL_DIR}/{}.voxels.pkl \
        --voxeldir=${VOXEL_DIR} \
        --tempdir=${FF_DIR} \
        --searchdir=${APO_DIR} \
        --featurize=${FEATURE_PROGRAM} \
        --numshell=6 \
        --width=1.25 \
        --overwrite


####################
#   CREATE LABELS  #
####################
cat ${LABELDLIST} | parallel -j 8 \
    python ${SCRIPT_HOME}/curate_dataset/label_voxels.py \
        --pdbid={} \
        --voxelfile=${VOXEL_DIR}/{}.voxels.pkl \
        --lmfile=${VOXEL_DIR}/{}.landmarks.pkl \
        --configfile=${SCRIPT_HOME}/curate_dataset/label_config.json \
        --hard \
        --outdir=${VOXEL_DIR} \
        --interactions HeavyAtomsite \
        --overwrite


# move intermediate output *.xyz to voxel detail folder
mv ${VOXEL_DIR}/*.xyz ${BACKUP_DIR}/

######################
#   Finish Curation  #
######################
cat ${LABELDLIST} | wc -l # 4516
ls ${VOXEL_DIR}/*.property.pvar | wc -l # 4516
ls ${VOXEL_DIR}/*.HeavyAtomsite.labels.pkl | wc -l # 4516, these three values should be consistent

echo "Finish creating ${PDBLIST}, ${LABELDLIST}, ${FF_DIR}, ${VOXEL_DIR}, ${BACKUP_DIR}"
echo "${FF_DIR} and ${BACKUP_DIR} can be deleted because the training doesn't depend on these files"
echo "But they may help to visualise and understand the curated dataset"
echo "Data files used for training are ${VOXEL_DIR}/*.property.pvar and ${VOXEL_DIR}/*._interactions_.labels.pkl"
