import os
import json
from utils.fewshot_path import few_shot_classifier, run_mizero, get_slide_feature
from dataset.wsi_datasets import WSIEmbeddingDataset
import torch
from torch.utils.data import DataLoader
import pandas as pd 
import numpy as np
import argparse
import logging

argparser = argparse.ArgumentParser()
argparser.add_argument('--model_name', type=str, default='resnet50')
argparser.add_argument('--feature_path', type=str, default=None)
argparser.add_argument('--task', type=str, default='camelyon')
argparser.add_argument('--seed', type=int, default=0)
argparser.add_argument('--extractor_path', type=str, default=None)
argparser.add_argument('--text_encoder_path', type=str, default=None)

args = argparser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(args.seed)


index_col = 'slide_id' # column with the slide ids
target_col = 'label' # column with the target labels
dir_path = args.feature_path
task = args.task # task name, used to load the correct prompts
model_name = args.model_name
os.makedirs('./results', exist_ok=True)

logging.basicConfig(filename=f'./results/{task}_{model_name}_{args.seed}.log', level=logging.INFO, filemode='a', format='%(asctime)s - %(message)s')

if task == 'camelyon':
    label_map = {'normal': 0, 'tumor': 1}
    df = pd.read_csv('./dataset_csv/camelyon.csv')
    prompt_file = './prompts/camelyon_prompts_all_per_class.json'

elif task == 'panda':
    label_map = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
    df = pd.read_csv('./dataset_csv/PANDA.csv')
    df = df.astype(str)
    prompt_file = './prompts/panda_prompts_all_per_class.json'

elif task == 'ubc-ocean':
    label_map = {'CC': 0, 'HGSC': 1, 'LGSC': 2, 'EC': 3, 'MC': 4}
    df = pd.read_csv('./dataset_csv/UBC-OCEAN.csv')
    df = df.astype(str)
    prompt_file = './prompts/ubc-ocean_prompts_all_per_class.json'

elif task == 'bcnb_er':
    label_map = {'Positive': 0, 'Negative': 1}
    df = pd.read_csv('./dataset_csv/BCNB_ER.csv')
    df = df.astype(str)
    prompt_file = './prompts/bcnb_er_prompts_all_per_class.json'

elif task == 'bcnb_pr':
    label_map = {'Positive': 0, 'Negative': 1}
    df = pd.read_csv('./dataset_csv/BCNB_PR.csv')
    df = df.astype(str)
    prompt_file = './prompts/bcnb_pr_prompts_all_per_class.json'

elif task == 'bcnb_her2':
    label_map = {'Positive': 0, 'Negative': 1}
    df = pd.read_csv('./dataset_csv/BCNB_HER2.csv')
    df = df.astype(str)
    prompt_file = './prompts/bcnb_her2_prompts_all_per_class.json'

    
df = df[df[target_col].isin(label_map.keys())].reset_index(drop=True)
logging.info(df['label'].value_counts())
sample_ks = [2**i for i in range(int(np.log2(df['label'].value_counts().min()))+1)]

results_list = []

for sample_k in sample_ks:
    sampled_df = df.groupby('label').sample(n=sample_k)
    remaining_df = df.drop(sampled_df.index)
    sampled_df = sampled_df.reset_index(drop=True)
    remaining_df = remaining_df.reset_index(drop=True)
    
    sampled_dataset = WSIEmbeddingDataset(df=sampled_df,
                                index_col=index_col,
                                target_col=target_col,
                                dir_path=dir_path,
                                model_name=model_name,
                                label_map=label_map)
    sampled_dataloader = DataLoader(sampled_dataset, 
                            batch_size=1, 
                            shuffle=False, 
                            num_workers=4)

    idx_to_class = {v:k for k,v in sampled_dataloader.dataset.label_map.items()}
    assert len(sampled_dataloader.dataset) == sample_k * len(label_map)
    print("num samples for creating slide_level_prototypes: ", len(sampled_dataloader.dataset))
    print(idx_to_class)

    with open(prompt_file) as f:
        prompts = json.load(f)['0']
    classnames = prompts['classnames']
    templates = prompts['templates']
    n_classes = len(classnames)
    classnames_text = [classnames[str(idx_to_class[idx])] for idx in range(n_classes)]
    for class_idx, classname in enumerate(classnames_text):
        print(f'{class_idx}: {classname}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    zeroshot_weights = few_shot_classifier(model_name = model_name, classnames = classnames_text, templates = templates, device=device, ckpt_path=args.text_encoder_path)
    zeroshot_weights = zeroshot_weights.to(device)

    all_features = get_slide_feature(zeroshot_weights, sampled_dataloader, device, model_name = model_name, ckpt_path=args.extractor_path)
    '''
    all_features: dict{cls: torch.tensor} where cls is the class index and torch.tensor is the feature tensor of shape (topj, n_features)
    '''
    slide_level_prototypes = []
    for i in range(n_classes):
        slide_level_prototypes.append(all_features[i].mean(0).unsqueeze(0))
    slide_level_prototypes = torch.cat(slide_level_prototypes, dim=0).T

    remaining_dataset = WSIEmbeddingDataset(df=remaining_df,
                                index_col=index_col,
                                target_col=target_col,
                                dir_path=dir_path,
                                model_name=model_name,
                                label_map=label_map)
    remaining_dataloader = DataLoader(remaining_dataset, 
                            batch_size=1, 
                            shuffle=False, 
                            num_workers=4)
    print("num samples for few shot: ", len(remaining_dataloader.dataset)) 

    results, dump = run_mizero(slide_level_prototypes, remaining_dataloader, device, 
                        dump_results=True, metrics=['bacc', 'weighted_f1', 'roc_auc'], model_name = model_name, ckpt_path=args.extractor_path)
    logging.info(f"sample_k: {sample_k}, results: {results}")
    results['sample_k'] = sample_k
    results_list.append(results)

# Convert results list to DataFrame and save as CSV
results_df = pd.DataFrame(results_list)
results_df.to_csv(f'./results/results_{task}_{model_name}_{args.seed}.csv', index=False)
logging.info("Results saved to CSV file: results_%s_%s_%d.csv", task, model_name, args.seed)