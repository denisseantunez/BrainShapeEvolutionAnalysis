#source PATH_TO_CONDA/conda.sh

#cd PATH_TO_PROJ_DIR
#conda activate ENV

#subjs=$(< mindboggle101_list.txt)

#for f in $subjs; 

for f in ../Surfaces/*.surf.gii; 
do 
    echo $f
    python spectral_align.py -r ../Surfaces/sub-001_species-Cercopithecus+cephus_hemi-L.surf.gii -s $f -o data/after_alignment --robust --verbose
done