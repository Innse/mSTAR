studies="Camelyon PANDA UBC-OCEAN BRCA_subtyping NSCLC RCC-DHMC"
ROOT_WSI="/path/to/wsi"
aggregator="/path/to/pretrained/aggregator.pth"
for study in $studies
do
    CUDA_VISIBLE_DEVICES=0 nohup python main.py --model ABMIL \
                                                      --study $study \
                                                      --root ${ROOT_WSI}/${study} \
                                                      --feature mSTAR \
                                                      --csv_file ./dataset_csv/${study}.csv \
                                                      --num_epoch 50 \
                                                      --batch_size 1 \
                                                      --lr 2e-4 \
                                                      --tqdm

    CUDA_VISIBLE_DEVICES=0 nohup python main.py --model TransMIL \
                                                      --study $study \
                                                      --root ${ROOT_WSI}/${study} \
                                                      --feature mSTAR \
                                                      --csv_file ./dataset_csv/${study}.csv \
                                                      --num_epoch 50 \
                                                      --batch_size 1 \
                                                      --lr 2e-4 \
                                                      --tqdm

done

echo "All jobs have been submitted."