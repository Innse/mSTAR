import os
import json
from utils.zeroshot_path import zero_shot_classifier, run_mizero
from dataset.wsi_datasets import WSIEmbeddingDataset
import torch
from torch.utils.data import DataLoader, RandomSampler
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import logging

argparser = argparse.ArgumentParser()
argparser.add_argument('--model_name', type=str, default='resnet50')
argparser.add_argument('--feature_path', type=str, default=None)
argparser.add_argument('--task', type=str, default='camelyon')
argparser.add_argument('--extractor_path', type=str, default=None)
argparser.add_argument('--text_encoder_path', type=str, default=None)
argparser.add_argument('--seed', type=int, default=42)

args = argparser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(args.seed)

index_col = 'slide_id'  # column with the slide ids
target_col = 'label'  # column with the target labels
dir_path = args.feature_path
task = args.task  # task name, used to load the correct prompts
model_name = args.model_name
os.makedirs('./results', exist_ok=True)

logging.basicConfig(filename=f'./results/{task}_{model_name}.log', level=logging.INFO, filemode='a', format='%(asctime)s - %(message)s')

logging.info("Starting the script with task: %s and model: %s", task, model_name)

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

dataset = WSIEmbeddingDataset(df=df,
                              index_col=index_col,
                              target_col=target_col,
                              dir_path=dir_path,
                              model_name=model_name,
                              label_map=label_map)
dataloader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=4)

idx_to_class = {v: k for k, v in dataloader.dataset.label_map.items()}
logging.info("Number of samples: %d", len(dataloader.dataset))
logging.info("Index to class mapping: %s", idx_to_class)

with open(prompt_file) as f:
    prompts = json.load(f)['0']
classnames = prompts['classnames']
templates = prompts['templates']
n_classes = len(classnames)
classnames_text = [classnames[str(idx_to_class[idx])] for idx in range(n_classes)]
for class_idx, classname in enumerate(classnames_text):
    logging.info('%d: %s', class_idx, classname)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info("Using device: %s", device)

zeroshot_weights = zero_shot_classifier(model_name=model_name, classnames=classnames_text, templates=templates, device=device, ckpt_path=args.text_encoder_path)
zeroshot_weights = zeroshot_weights.to(device)

logging.info('Bootstrapping...')

n_iterations = 1000
all_bacc = {k: [] for k in [1, 5, 10, 50, 100]}
all_weighted_f1 = {k: [] for k in [1, 5, 10, 50, 100]}
all_roc_auc = {k: [] for k in [1, 5, 10, 50, 100]}
num_samples = len(dataset)
for i in tqdm(range(n_iterations)):
    sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples)
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            sampler=sampler,
                            num_workers=4)
    results, dump = run_mizero(zeroshot_weights, dataloader, device,
                               dump_results=True, metrics=['bacc', 'weighted_f1', 'roc_auc'],
                               model_name=model_name, ckpt_path=args.extractor_path)
    for k in [1, 5, 10, 50, 100]:
        all_bacc[k].append(results['bacc'][k])
        all_weighted_f1[k].append(results['weighted_f1'][k])
        all_roc_auc[k].append(results['roc_auc'][k])
    print(f"bacc: {results['bacc']}, weighted_f1: {results['weighted_f1']}, roc_auc: {results['roc_auc']}")
# Function to format the confidence interval display
def format_ci(mean, std):
    lower = mean - 1.96 * std
    upper = mean + 1.96 * std
    return f"{mean:.4f} ({lower:.4f}, {upper:.4f})"

# Calculating and formatting the 95% confidence intervals for each metric
formatted_ci_95_bacc = {k: format_ci(np.mean(all_bacc[k]), np.std(all_bacc[k], ddof=1)) for k in [1, 5, 10, 50, 100]}
formatted_ci_95_weighted_f1 = {k: format_ci(np.mean(all_weighted_f1[k]), np.std(all_weighted_f1[k], ddof=1)) for k in [1, 5, 10, 50, 100]}
formatted_ci_95_roc_auc = {k: format_ci(np.mean(all_roc_auc[k]), np.std(all_roc_auc[k], ddof=1)) for k in [1, 5, 10, 50, 100]}

# Logging formatted results
logging.info("BACC 95%% CI: %s", formatted_ci_95_bacc)
logging.info("Weighted F1 95%% CI: %s", formatted_ci_95_weighted_f1)
logging.info("ROC AUC 95%% CI: %s", formatted_ci_95_roc_auc)

results_df = pd.DataFrame({
    'k': [1, 5, 10, 50, 100],
    'bacc': [formatted_ci_95_bacc[k] for k in [1, 5, 10, 50, 100]],
    'weighted_f1': [formatted_ci_95_weighted_f1[k] for k in [1, 5, 10, 50, 100]],
    'roc_auc': [formatted_ci_95_roc_auc[k] for k in [1, 5, 10, 50, 100]]
})

results_df.to_csv(f'./results/results_{task}_{model_name}_{args.seed}.csv', index=False)
logging.info("Results saved to CSV file: results_%s_%s_%d.csv", task, model_name, args.seed)
