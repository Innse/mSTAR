# Feature extraction
Here we provide detailed instructions on how to use our mSTAR to conduct feature extraction on WSIs.

Before we start, make sure you have installed the required dependencies mentioned [here](https://github.com/Innse/mSTAR/blob/main/README.md)

Some of the most important libraries are:
```
timm==0.9.8
openslide==3.4.1
openslide-python==1.3.1
```
## Pipeline
1. Prepare your WSIs either from your private sources or public sources like [TCGA](https://www.cancer.gov/ccg/research/genome-sequencing/tcga). Download the weights from our [Huggingface model page](https://huggingface.co/Wangyh/mSTAR) and we recommand to put the weights file in [models/ckpts](models/ckpts)

2. Follow the instruction from [CLAM](https://github.com/mahmoodlab/CLAM) to preprocess the whole slide images which includes tissue segmentation and stitching. After preprocessing you can get a coordinator folder like this
```bash
DIR_TO_COORDS
    └──patches/
        ├── slide_1.h5
        ├── slide_2.h5
        ├── slide_3.h5
        └── ...
```
3. Run the following command to conduct feature extraction on WSIs
```bash
models='mSTAR'
declare -A gpus
gpus['mSTAR']=0

CSV_FILE_NAME="./dataset_csv/example.csv"

DIR_TO_COORDS="./example_coords"
DATA_DIRECTORY="./example_slide"

FEATURES_DIRECTORY="./output/example"

ext=".jpg"
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
                --slide_ext $ext
done
``` 
This command will conduct feature extraction on the example WSI we provide. For your own dataset, you need to specify the values for ```CSV_FILE_NAME```, ```DIR_TO_COORDS``` and ```FEATURES_DIRECTORY```.

Here is a detailed description of the command line arguments.
- DATA_DIRECTORY: This should be set to the directory which contains the WSI data.
- DIR_TO_COORDS: This should be set to the directory that contains the coordinate information for the WSI patches preprocessed through [CLAM](https://github.com/mahmoodlab/CLAM).
- FEATURES_DIRECTORY: This is the directory where you want to store the extracted features. 
- model: The feature extractor you want to use, here we use our mSTAR.
- slide_ext: The WSI file format. For TCGA data, it is .svs, and for the provided example, it is .jpg.

If you run the command above, you will get a folder like this:
```bash
output
  └─example
    └─pt_files
      └─mSTAR
        ├── example_1.pt
        ├── example_2.pt
        └── example_3.pt
```