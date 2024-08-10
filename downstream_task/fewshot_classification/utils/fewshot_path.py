import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .utils import AverageMeter, merge_dict
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score, 
                             classification_report, roc_auc_score)


def tokenize(tokenizer, texts):
    # model context length is 128, but last token is reserved for <cls>
    # so we use 127 and insert <pad> at the end as a temporary placeholder
    tokens = tokenizer.batch_encode_plus(texts, 
                                        max_length = 127,
                                        add_special_tokens=True, 
                                        return_token_type_ids=False,
                                        truncation = True,
                                        padding = 'max_length',
                                        return_tensors = 'pt')
    tokens = F.pad(tokens['input_ids'], (0, 1), value=tokenizer.pad_token_id)
    return tokens



@torch.no_grad()
def few_shot_classifier(model_name, classnames, templates, device=None, ckpt_path=None):
    """
    classnames: list of lists of classnames (one list of classnames per class)
    templates: list of templates 
    """

    if model_name == 'conch':
        from models.open_clip_custom import create_model_from_pretrained, get_tokenizer
        model, _ = create_model_from_pretrained(model_cfg='conch_ViT-B-16', checkpoint_path='/home/ywangrm/Zero_shot/classification/models/ckpts/conch.pth', device=device)
        model.eval()
        tokenizer = get_tokenizer()

        zeroshot_weights = []
        for classnames_for_class in classnames:
            embeddings_for_class = []
            for classname in classnames_for_class:
                texts = [template.replace('CLASSNAME', classname) for template in templates]
                token_ids = tokenize(tokenizer, texts) # Tokenize with custom tokenizer
                token_ids = token_ids.to(device)
                classname_embeddings = model.encode_text(token_ids)
                # classname_embeddings: [num_templates, embedding_dim]
                embeddings_for_class.append(F.normalize(classname_embeddings, dim=-1))
            
            class_embedding = torch.stack(embeddings_for_class, dim=0)
            # over all templates and classnames
            class_embedding = class_embedding.mean(dim=(0, 1))
            class_embedding /= class_embedding.norm()

    # class_embedding: [embedding_dim]
            zeroshot_weights.append(class_embedding)

    elif model_name == 'plip':
        from models.plip import PLIP
        model = PLIP('vinid/plip')

        zeroshot_weights = []
        for classnames_for_class in classnames:
            embeddings_for_class = []
            for classname in classnames_for_class:
                texts = [template.replace('CLASSNAME', classname) for template in templates]
                # print(texts)
                classname_embeddings = model.encode_text(texts, batch_size=1)
                # print(classname_embeddings.shape)
                # classname_embeddings: [num_templates, embedding_dim]
                classname_embeddings = torch.tensor(classname_embeddings)
                embeddings_for_class.append(F.normalize(classname_embeddings, dim=-1))
            # class_embedding: [num_classnames, num_templates, embedding_dim]
            class_embedding = torch.stack(embeddings_for_class, dim=0)
            # over all templates and classnames
            class_embedding = class_embedding.mean(dim=(0, 1))
            class_embedding /= class_embedding.norm()

    # class_embedding: [embedding_dim]
            zeroshot_weights.append(class_embedding)
    
    elif  model_name in ['resnet50', 'uni', 'mSTAR']:
        from transformers import AutoTokenizer
        from models.mSTAR import Text_encoder
        print("Using mSTAR's text encoder")
        model = Text_encoder(config_text={'model_name_or_path':'dmis-lab/biobert-base-cased-v1.2'})
        model.to(device)
        model.eval()
        print('loading model from checkpoint', ckpt_path)

        checkpoint = torch.load(ckpt_path,map_location='cpu')
        # print((checkpoint['model_state_dict']).keys())
        text_encoder_param = {k.replace('module.',''):v for k,v in checkpoint['model_state_dict'].items() if 'text_encoder' in k}
        additional_params = {k.replace('module.',''):v for k,v in checkpoint['model_state_dict'].items() if 'proj_text' in k}
    
        # Use the update() method to add the additional_params to text_encoder_param
        text_encoder_param.update(additional_params)

        msg = model.load_state_dict(text_encoder_param, strict=False)
        if msg.missing_keys:
            print(f"missing keys: {msg.missing_keys}")
        elif msg.unexpected_keys:
            print(f"unexpected keys: {msg.unexpected_keys}")
        else:
            print("load text encoder successfully")
        tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")

        zeroshot_weights = []
        for classnames_for_class in classnames:
            embeddings_for_class = []
            for classname in classnames_for_class:
                texts = [template.replace('CLASSNAME', classname) for template in templates]
                token_ids = tokenizer(texts, padding='max_length', truncation=True, return_tensors="pt", max_length=512)
                input_ids = token_ids['input_ids'].to(device)
                attention_mask = token_ids['attention_mask'].to(device)
                # print(texts)
                classname_embeddings = model(input_ids.squeeze(1), attention_mask.squeeze(1))
                # print(classname_embeddings.shape)
                # classname_embeddings: [num_templates, embedding_dim]
                classname_embeddings = torch.tensor(classname_embeddings)
                embeddings_for_class.append(F.normalize(classname_embeddings, dim=-1))
            # class_embedding: [num_classnames, num_templates, embedding_dim]
            class_embedding = torch.stack(embeddings_for_class, dim=0)
            # over all templates and classnames
            class_embedding = class_embedding.mean(dim=(0, 1))
            class_embedding /= class_embedding.norm()

    # class_embedding: [embedding_dim]
            zeroshot_weights.append(class_embedding)

    # zeroshot_weights: [embedding_dim, num_classes]
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    return zeroshot_weights

def topj_pooling(logits, topj):
    """
    logits: N x C logit for each patch
    topj: tuple of the top number of patches to use for pooling
    """
    # Sums logits across topj patches for each class, to get class prediction for each topj
    maxj = min(max(topj), logits.size(0)) # Ensures j is smaller than number of patches. Unlikely for number of patches to be < 10, but just in case
    values, indices = logits.topk(maxj, 0, True, True) # maxj x C
    
    # Calculate the mean of the top j logits
    preds = {j: values[:min(j, maxj)].mean(dim=0, keepdim=True) for j in topj} # dict of 1 x C logit scores
    pooled_logits = {key: val for key, val in preds.items()}
    
    # Get the predicted class indices
    preds = {key: val.argmax(dim=1) for key, val in preds.items()} # dict of predicted class indices
    
    # Prepare the top j logits and their indices
    topj_logits_indices = {j: (values[:min(j, maxj)], indices[:min(j, maxj)]) for j in topj}
    
    return preds, pooled_logits, topj_logits_indices

@torch.no_grad()
def get_slide_feature(classifier, dataloader, device, topj = (5,10,50,100), model_name=None, ckpt_path=None):                                            
    all_logits = {}
    all_features = {}
    for idx, data in enumerate((dataloader)): # batch size is always 1, 
        image_features = data['img'].to(device).squeeze(0)
        label = data['label'][0].item()
        if label not in all_logits:
            all_logits[label] = []
            all_features[label] = []
        if model_name in ['resnet50', 'uni', 'mSTAR']:
            proj = nn.Linear(1024, 512).to(device)
            ckpt = torch.load(ckpt_path, map_location=device)
            ckpt = ckpt['model_state_dict']
            updated_ckpt = {}
            for k, v in ckpt.items():
                k = k.replace('module.', '')
                if k in ['proj.weight', 'proj.bias']:
                    k = k.replace('proj.', '')
                    updated_ckpt[k] = v
            msg = proj.load_state_dict(updated_ckpt)
            print('loading proj:',msg)
            image_features = proj(image_features)
        image_features = F.normalize(image_features, dim=-1) 
        logits = image_features @ classifier
        # print(logits.shape)
        all_logits[label].append(logits)
        all_features[label].append(image_features)
    for k in all_logits:
        all_logits[k] = torch.cat(all_logits[k], dim=0)
        all_features[k] = torch.cat(all_features[k], dim=0)
    
    # print(all_logits)
    for k in all_logits:
        maxj = min(max(topj), all_logits[k].size(0)) 
        _, indices = torch.topk(all_logits[k], maxj, dim=0)
        indices = indices[:,k]
        all_features[k] = all_features[k][indices] 
    return all_features


@torch.no_grad()
def get_slide_feature_aggregator(aggregator, dataloader, device):                                            
    all_features = {}
    for idx, data in enumerate((dataloader)): # batch size is always 1, 
        image_features = data['img'].to(device).squeeze(0)
        label = data['label'][0].item()
        image_features = aggregator(image_features)
        if label not in all_features:
            all_features[label] = []
        image_features = F.normalize(image_features, dim=-1) 
        # print(logits.shape)
        all_features[label].append(image_features)
    for k in all_features:
        all_features[k] = torch.cat(all_features[k], dim=0)
    
    return all_features




@torch.no_grad()
def run_mizero(classifier, dataloader, device, topj=(1, 5, 10, 50, 100), 
               dump_results=False, dump_patch_level=False, 
               metrics=['acc', 'bacc', 'weighted_kappa', 'kappa', 'roc_auc', 'weighted_f1'], 
               model_name=None, ckpt_path=None):
        
    dict_keys = list(topj)
    meters = {j: AverageMeter() for j in dict_keys}

    logits_all, targets_all, patch_logits_all, coords_all, preds_all = {}, [], [], [], {}
    correct_ids = []  

    if model_name in ['resnet50', 'uni', 'mSTAR']:
        proj = nn.Linear(1024, 512).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        ckpt = ckpt['model_state_dict']
        updated_ckpt = {}
        for k, v in ckpt.items():
            k = k.replace('module.', '')
            if k in ['proj.weight', 'proj.bias']:
                k = k.replace('proj.', '')
                updated_ckpt[k] = v
        msg = proj.load_state_dict(updated_ckpt)
        print('loading proj:', msg)
    
    for idx, data in enumerate(dataloader):  # batch size is always 1, 
        image_features = data['img'].to(device).squeeze(0)
        target = data['label'].to(device)
        coords = data['coords']
        sample_id = data['slide_id'][0]  
        if not isinstance(coords, list):
            coords = coords.squeeze(0).numpy()
        coords_all.append(coords)

        if model_name in ['resnet50', 'uni', 'mSTAR']:
            image_features = proj(image_features) 
        image_features = F.normalize(image_features, dim=-1) 
        logits = image_features @ classifier
        if dump_results and dump_patch_level:
            patch_logits_all.append(logits.cpu().numpy())

        preds, pooled_logits, _ = topj_pooling(logits, topj=topj)
        results = {key: (val == target).float().item() for key, val in preds.items()}
        
        preds_all = merge_dict(preds_all, preds, value_fn=lambda x: x.item())
        logits_all = merge_dict(logits_all, pooled_logits, value_fn=lambda x: x.cpu().numpy())
        targets_all.append(target.cpu().numpy())

        for j in topj:
            if results[j] == 1.0:  # 如果预测正确
                correct_ids.append(sample_id)
            meters[j].update(results[j], n=1)  # Update AverageMeters with new results

    # Save raw preds & targets
    targets_all = np.concatenate(targets_all, axis=0)
    logits_all = {key: np.concatenate(logits_all[key], axis=0) for key in dict_keys}
    probs_all = {key: F.softmax(torch.from_numpy(logits_all[key]) * nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp().item(), dim=1).numpy() for key in dict_keys}
    preds_all = {key: np.array(preds_all[key]) for key in dict_keys}
    baccs = {key: balanced_accuracy_score(targets_all, val) for key, val in preds_all.items()}
    cls_rep = {key: classification_report(targets_all, val, output_dict=True, zero_division=0) for key, val in preds_all.items()}
    kappas = {key: cohen_kappa_score(targets_all, val) for key, val in preds_all.items()}
    weighted_kappas = {key: cohen_kappa_score(targets_all, val, weights='quadratic') for key, val in preds_all.items()}
    roc_aucs = {}
    for key, probs in probs_all.items():
        n_classes = probs.shape[1]
        if n_classes == 2:
            class_probs = probs[:, 1]
            roc_kwargs = {}
        else:
            class_probs = probs
            roc_kwargs = {'multi_class': 'ovo', 'average': 'macro'}        
        try:
            roc_auc = roc_auc_score(targets_all, class_probs, **roc_kwargs)
        except ValueError:
            roc_auc = np.nan
        roc_aucs[key] = roc_auc

    accs = {j: meters[j].avg for j in topj}

    dump = {}
    results = {'acc': accs, 
               'bacc': baccs, 
               'report': cls_rep, 
               'kappa': kappas,
               'weighted_kappa': weighted_kappas, 
               'roc_auc': roc_aucs,
               'weighted_f1': {key: cls_rep[key]['weighted avg']['f1-score'] for key in dict_keys}}
    results = {k: results[k] for k in metrics}
    if dump_results:
        dump['logits'] = logits_all
        dump['targets'] = targets_all
        dump['preds'] = preds_all
        dump['correct_ids'] = correct_ids  # 保存预测正确的ID
        dump['temp_scale'] = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp().item()
        
        if dump_patch_level: 
            dump['patch_logits'] = patch_logits_all
            dump['coords'] = coords_all

    return results, dump
