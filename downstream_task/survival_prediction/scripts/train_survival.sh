feature_path='/feature_path'

studies='LUSC'

models='AttMIL'
# models='TransMIL'


features='mSTAR'

lr=2e-4
# ckpt for pretrained aggregator
# aggregator='aggregator'

# export WANDB_MODE=dryrun

cd ..

for feature in $features
do
    for study in $studies
    do
        for model in $models
        do
            CUDA_VISIBLE_DEVICES=0 python main.py --model $model \
                                                --csv_file ./dataset_csv/${study}_Splits.csv \
                                                --feature_path $feature_path \
                                                --study $study \
                                                --modal WSI \
                                                --num_epoch 30 \
                                                --batch_size 1 \
                                                --lr $lr \
                                                --feature $feature \
                                                # --aggregator $aggregator
        done
    done
done

