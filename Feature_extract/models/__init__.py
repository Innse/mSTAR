import torch
import numpy as np


__all__ = ['list_models', 'get_model', 'get_custom_transformer']


__implemented_models = {
    'resnet50': 'image-net',
    'plip': 'https://huggingface.co/vinid/plip',
    'uni': 'https://huggingface.co/MahmoodLab/UNI',
    'conch': 'https://huggingface.co/MahmoodLab/CONCH',
    'mstar': 'https://huggingface.co/Wangyh/mSTAR'
    
}


def list_models():
    print('The following are implemented models:')
    for k, v in __implemented_models.items():
        print('{}: {}'.format(k, v))
    return __implemented_models


def get_model(model_name, device, gpu_num, model_checkpoint=None):
    """_summary_

    Args:
        model_name (str): the name of the requried model
        device (torch.device): device, e.g. 'cuda'
        gpu_num (int): the number of GPUs used in extracting features

    Raises:
        NotImplementedError: if the model name does not exist

    Returns:
        nn.Module: model
    """
    if model_name == 'resnet50':
        from models.resnet_custom import resnet50_baseline
        model = resnet50_baseline(pretrained=True).to(device)
    elif model_name == 'plip':
        from models.plip import plip
        model = plip(device, gpu_num)
    elif model_name.lower() == 'uni':
        from models.uni import get_uni_model
        model = get_uni_model(device)
    elif model_name.lower() == 'conch':
        from models.Conch import get_conch_model
        model = get_conch_model(device=device)
    elif model_name.lower() == 'mstar':
        from models.mSTAR import get_mSTAR_model
        model = get_mSTAR_model(device)
    else:
        raise NotImplementedError(f'{model_name} is not implemented')

    if model_name == 'resnet50':
        if gpu_num > 1:
            model = torch.nn.parallel.DataParallel(model)
        model = model.eval()
    return model


def get_custom_transformer(model_name):
    """_summary_

    Args:
        model_name (str): the name of model

    Raises:
        NotImplementedError: not implementated

    Returns:
        torchvision.transformers: the transformers used to preprocess the image
    """
    if model_name == 'resnet50':
        from models.resnet_custom import custom_transforms
        custom_trans = custom_transforms()
    elif model_name == 'plip':
        from torchvision import transforms as tt
        custom_trans = tt.Lambda(lambda x: torch.from_numpy(np.array(x)))
    elif model_name.lower() == 'uni':
        from models.uni import get_uni_trans
        custom_trans = get_uni_trans()
    elif model_name.lower() == 'conch':
        from models.Conch import get_conch_trans
        custom_trans = get_conch_trans()
    elif model_name.lower() == 'mstar':
        from models.mSTAR import get_mSTAR_trans
        custom_trans = get_mSTAR_trans()

    else:
        raise NotImplementedError('Transformers for {} is not implemented ...'.format(model_name))

    return custom_trans
