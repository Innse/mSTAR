studies="BRCA CRC GBMLGG HNSC KIRC LUAD LUSC SKCM UCEC"
ROOT_WSI="/path/to/wsi"
ROOT_OMIC="/path/to/omic"

for study in $studies
do
    CUDA_VISIBLE_DEVICES=0 python main.py --model MCAT \
                                          --root_path ${ROOT_WSI}/${study} \
                                          --root_omic ${ROOT_OMIC}/${study} \
                                          --excel_file ./dataset_csv/${study}_Splits.csv \
                                          --modal WSI_Gene \
                                          --num_epoch 30 \
                                          --signatures ./dataset_csv/signatures.csv \
                                          --batch_size 1 \
                                          --tqdm

    CUDA_VISIBLE_DEVICES=0 python main.py --model MOTCAT \
                                          --root_path ${ROOT_WSI}/${study} \
                                          --root_omic ${ROOT_OMIC}/${study} \
                                          --excel_file ./dataset_csv/${study}_Splits.csv \
                                          --modal WSI_Gene \
                                          --num_epoch 30 \
                                          --signatures ./dataset_csv/signatures.csv \
                                          --batch_size 1 \
                                          --tqdm

    CUDA_VISIBLE_DEVICES=0 python main.py --model CMTA \
                                          --root_path ${ROOT_WSI}/${study} \
                                          --root_omic ${ROOT_OMIC}/${study} \
                                          --excel_file ./dataset_csv/${study}_Splits.csv \
                                          --modal WSI_Gene \
                                          --num_epoch 30 \
                                          --signatures ./dataset_csv/signatures.csv \
                                          --loss nll_surv_l1 \
                                          --batch_size 1 \
                                          --tqdm

    CUDA_VISIBLE_DEVICES=0 python main.py --model Porpoise \
                                          --root_path ${ROOT_WSI}/${study} \
                                          --root_omic ${ROOT_OMIC}/${study} \
                                          --excel_file ./dataset_csv/${study}_Splits.csv \
                                          --modal WSI_Gene \
                                          --num_epoch 30 \
                                          --batch_size 1 \
                                          --tqdm
done

echo "All jobs have been submitted."