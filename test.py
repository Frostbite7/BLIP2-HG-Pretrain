from .blip_2_m import Blip2QFormerModel
from transformers import Blip2QFormerConfig
import torch

config = Blip2QFormerConfig()
print(config.query_length)

qformer = Blip2QFormerModel(config, 32)
mask_test = torch.Tensor([[1, 1, 0], [1, 0, 0]])
print(qformer.invert_attention_mask(mask_test))