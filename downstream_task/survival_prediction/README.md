# Survival Prediction
Here we provide detailed instruction on how to run survival prediction using our scripts.

Before we strat, make sure you have installed the required dependencies mentioned [here](https://github.com/Innse/mSTAR/blob/main/README.md) and have already extracted the mSTAR features from your datasets.

Take [TCGA-LUSC]((https://portal.gdc.cancer.gov/projects/TCGA-LUSC)) as an example.
The feature directory should look like:
```
TCGA-LUSC
  └─pt_files
      └─mSTAR
        ├── feature_1.pt
        ├── feature_2.pt
        ├── feature_3.pt
        └── ...

```
We provided the [split file](dataset_csv/LUSC_Splits.csv) for reproduction. To run the scripts, using the following commands
```bash
conda activate mSTAR
cd downstream_task/survival_prediction/scripts
```
In [train_survival.sh](scripts/train_survival.sh), you need to specify the path of the feature directory and choose the model.
```bash
feature_path='/feature_path' #change here
studies='LUSC'

models='AttMIL'
# models='TransMIL'
# models='TransMIL_Pre'
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

```
Here is a detailed description of the command line arguments:
- model: The model you want to use. Currently, we support AttMIL, TransMIL, and pretrained TransMIL.
- csv_file: This should be set to the CSV file that contains the dataset splits (train/validation/test).
- feature_path: This should be set to the directory that contains the extracted features.
- study: This is the dataset you are currently using.
- aggregator: If you want to use a pretrained TransMIL model, this should be set to the path of the model's weights.

After you have completed all the settings, you can launch the script.
```bash
./train_survival.sh
```

The total time to run this demo may take around **10** mins for AttMIL and **15** mins for TransMIL and pretrained TransMIL.