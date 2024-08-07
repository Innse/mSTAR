
# model="resnet50"
# models='plip'
# models='uni'
# models='conch'
models='mSTAR'
declare -A gpus

gpus['resnet50']=0
gpus['conch']=0
gpus['uni']=0
gpus['mSTAR']=0

CSV_FILE_NAME="./dataset_csv/LUAD.csv"

DIR_TO_COORDS="your own DIR_TO_COORDS"
DATA_DIRECTORY="your own DATA_DIRECTORY"

FEATURES_DIRECTORY="where you what to store the features"

datatype="tcga" # extra path process for TCGA dataset, direct mode do not care use extra path
ext=".svs"
for model in $models
do
        echo $model", GPU is:"${gpus[$model]}
        export CUDA_VISIBLE_DEVICES=${gpus[$model]}

        python extract_feature.py \
                --data_h5_dir $DIR_TO_COORDS \
                --data_slide_dir $DATA_DIRECTORY \
                --csv_path $CSV_FILE_NAME \
                --feat_dir $FEATURES_DIRECTORY \
                --batch_size 256 \
                --model $model \
                --datatype $datatype \
                --slide_ext $ext
done
