PROJ_HOME=FeatureDock
DATA_DIR=${PROJ_HOME}/data
PDBBIND_DIR=${DATA_DIR}/refined-set
PDBLIST=${DATA_DIR}/pdblist.txt
mkdir -p ${DATA_DIR}

####################################
#   DOWNLOAD PDBBIND REFINED SET   #
####################################
mkdir -p ${DATA_DIR}
echo "Download PDBBind v2020 refined dataset (Please login PDBBind before downloading if necessary)"
echo "See introduction to PDBBind dataset: http://www.pdbbind.org.cn/download/pdbbind_2020_intro.pdf"
if [ ! -e ${DATA_DIR}/PDBbind_v2020_refined.tar.gz ]; then
    wget http://www.pdbbind.org.cn/download/PDBbind_v2020_refined.tar.gz \
        -O ${DATA_DIR}/PDBbind_v2020_refined.tar.gz
    echo "Finish downloading PDBbind_v2020_refined.tar.gz"
fi
tar zxvf ${DATA_DIR}/PDBbind_v2020_refined.tar.gz -C ${DATA_DIR}
echo "${DATA_DIR}/PDBbind_v2020_refined.tar.gz has been extracted to ${PDBBIND_DIR}"


#############################
#   OUTPUT STRUCTURES IDS   #
#############################
ls ${PDBBIND_DIR}/*/*_protein.pdb | xargs basename -a | cut -d'_' -f1 > ${PDBLIST}

# Print number of structures
cat ${PDBLIST} | wc -l # 5316