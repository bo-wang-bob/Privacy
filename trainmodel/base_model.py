import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
    
    def freeze_backbone(self):
        raise NotImplementedError("freeze_backbone方法需在子类中实现")
    
    def unfreeze_backbone(self):
        raise NotImplementedError("unfreeze_backbone方法需在子类中实现")
    
    def freeze_head(self):
        raise NotImplementedError("freeze_head方法需在子类中实现")
    
    def unfreeze_head(self):
        raise NotImplementedError("unfreeze_head方法需在子类中实现")
    
    def forward_backbone(self, x):
        raise NotImplementedError("forward_backbone方法需在子类中实现")
    
    def forward(self, x, return_intermediate=False):
        raise NotImplementedError("forward方法需在子类中实现")

    def get_token_attention(self) -> torch.Tensor:
        raise NotImplementedError("get_token_attention方法需在子类中实现")

    def get_text_features(self, normalize: bool = True) -> torch.Tensor:
        raise NotImplementedError("get_text_features方法需在子类中实现")
