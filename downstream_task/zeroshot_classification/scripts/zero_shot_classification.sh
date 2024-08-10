declare -A task_feature_paths
task_feature_paths=( 
    ["panda"]='/path/to/PANDA' 
    ["camelyon"]='/path/to/CAMELYON' 
    ["ubc-ocean"]='/path/to/UBC-OCEAN' 
    ["bcnb_her2"]='/path/to/BCNB' 
    ["bcnb_er"]='/path/to/BCNB' 
    ["bcnb_pr"]='/path/to/BCNB' 
)



tasks='camelyon ubc-ocean bcnb_er bcnb_her2 bcnb_pr panda'
model_names='resnet50 uni conch plip mSTAR'

extractor_path='/path/to/extractor'
text_encoder_path='/path/to/textencoder'

export CUDA_VISIBLE_DEVICES=0
cd ..


for task in $tasks
do
    for model_name in $model_names
    do
        feature_path=${task_feature_paths[$task]}
        
        python main.py \
            --task $task \
            --model_name $model_name \
            --feature_path $feature_path \
            --extractor_path $extractor_path \
            --text_encoder_path $text_encoder_path
    done
done