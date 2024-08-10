import timm
import torch

from torch import nn
from torchvision import transforms
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

    
    
def get_mSTAR_trans():
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transform



def get_mSTAR_model(device):
    model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
    )
    model.load_state_dict(torch.load('./models/ckpts/mSTAR.pth', map_location="cpu"), strict=True)
    model.eval()
    
    return model.to(device)

class Text_encoder(nn.Module):
    def __init__(self, config_text):
        super().__init__()
        self.config_text = config_text
        config = AutoConfig.from_pretrained(
                self.config_text['model_name_or_path']
                )
        self.text_encoder = AutoModelForMaskedLM.from_pretrained(
            self.config_text['model_name_or_path'],
            config = config
        )
        self.proj_text = nn.Linear(768, 512)
        
    
    def forward(self, reports, attention_mask):
        text_logits = self.text_encoder.bert(input_ids=reports, attention_mask=attention_mask)[0][:, 0, :]
        # outputs['cls_text'] = F.normalize(self.proj_text(text_logits), p=2, dim=1)
        outputs = self.proj_text(text_logits)
        return outputs