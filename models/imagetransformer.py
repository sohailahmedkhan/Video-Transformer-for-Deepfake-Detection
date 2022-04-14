"""model.py - Model and module class for ViT.
   They are built to mirror those in the official Jax implementation.
"""

from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
import timm
from torch.utils import model_zoo

from vit_configs.configs import PRETRAINED_MODELS, load_pretrained_weights, as_tuple, resize_positional_embedding_
from transformer import *

# Import 3DDFA Packages
from models.DDFA import *


class Transformer(nn.Module):
    """Transformer with Self-Attentive Blocks"""
    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask)
        return x
    

class PositionalEmbedding1D(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, seq_len, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))
    
    def forward(self, x):
        """Input has shape `(batch_size, seq_len, emb_dim)`"""
        return x + self.pos_embedding
    
class SequenceEmbedding(nn.Module):
    """Adds (optionally learned) sequence embeddings to the inputs."""

    def __init__(self, seq_len, dim):
        super().__init__()
        self.seq_embedding = nn.Parameter(torch.zeros(1, 162, dim))
    
    def forward(self, x):
        """Input has shape `(batch_size, seq_len, emb_dim)`"""
        return x + self.seq_embedding    


class ImageTransformer(nn.Module):
    """
    Args:
        name (str): Model name, e.g. 'B_16'
        pretrained (bool): Load pretrained weights
        in_channels (int): Number of channels in input data
        num_classes (int): Number of classes, default 1000

    References:
        [1] https://openreview.net/forum?id=YicbFdNTTy
    """

    def __init__(
        self, 
        name: Optional[str] = None, 
        pretrained: bool = False, 
        patches: int = 16,
        dim: int = 768,
        ff_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        attention_dropout_rate: float = 0.0,
        dropout_rate: float = 0.1,
        representation_size: Optional[int] = None,
        load_repr_layer: bool = False,
        classifier: str = 'token',
        positional_embedding: str = '1d',
        in_channels: int = 3, 
        image_size: Optional[int] = None,
        num_classes: Optional[int] = None,
        seq_embed: bool = True, 
        hybrid: bool = True,
        device: str = 'cpu'
    ):
        super().__init__()

        # Configuration
        if name is None:
            check_msg = 'must specify name of pretrained model'
            assert not pretrained, check_msg
            assert not resize_positional_embedding, check_msg
            if num_classes is None:
                num_classes = 1000
            if image_size is None:
                image_size = 384
        else:  # load pretrained model
            assert name in PRETRAINED_MODELS.keys(), \
                'name should be in: ' + ', '.join(PRETRAINED_MODELS.keys())
            config = PRETRAINED_MODELS[name]['config']
            patches = config['patches']
            dim = config['dim']
            ff_dim = config['ff_dim']
            num_heads = config['num_heads']
            num_layers = config['num_layers']
            attention_dropout_rate = config['attention_dropout_rate']
            dropout_rate = config['dropout_rate']
            representation_size = config['representation_size']
            classifier = config['classifier']
            if image_size is None:
                image_size = PRETRAINED_MODELS[name]['image_size']
            if num_classes is None:
                num_classes = PRETRAINED_MODELS[name]['num_classes']
        self.seq_embed=seq_embed
        self.hybrid=hybrid
        self.image_size = image_size
        self.device = device
        print(device)         
#         print(hybrid)
        # Image and patch sizes
        h, w = as_tuple(image_size)  # image sizes
        fh, fw = as_tuple(patches)  # patch sizes
        gh, gw = h // fh, w // fw  # number of patches
        seq_len = gh * gw
        
        if self.hybrid == False:
            # Patch embedding
            self.patch_embedding = nn.Conv2d(in_channels, dim, kernel_size=(16,32), stride=(16, 32))
        else:
            #Get CNN feature extractor
            self.xception_feature_extractor = timm.create_model('xception', features_only=True, pretrained=True).eval()            
            
        self.sequence_embedding = SequenceEmbedding(162, dim)
        self.linear_1 = torch.nn.Linear(100, 162)
        self.proj = nn.Conv2d(2048, 768, 1)
                
#       Class token
        if classifier == 'token':
            self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
            seq_len += 1
        
#       Positional embedding
        if positional_embedding.lower() == '1d':
            self.positional_embedding = PositionalEmbedding1D(seq_len, dim)
        else:
            raise NotImplementedError()
                    
        #Transformer
        self.transformer = Transformer(num_layers=num_layers, dim=dim, num_heads=num_heads, 
                                       ff_dim=ff_dim, dropout=dropout_rate)
        
        # Representation layer
        if representation_size and load_repr_layer:
            self.pre_logits = nn.Linear(dim, representation_size)
            pre_logits_size = representation_size
        else:
            pre_logits_size = dim

        # Classifier head
        self.norm = nn.LayerNorm(pre_logits_size, eps=1e-6)
        self.fc = nn.Linear(pre_logits_size, num_classes)

        # Initialize weights
        self.init_weights()
        
        # Load pretrained model
        if pretrained:
            pretrained_num_channels = 3
            pretrained_num_classes = PRETRAINED_MODELS[name]['num_classes']
            pretrained_image_size = PRETRAINED_MODELS[name]['image_size']
            load_pretrained_weights(
                self, name, 
                load_fc=(num_classes == pretrained_num_classes),
                load_repr_layer=load_repr_layer,
                resize_positional_embedding=(image_size != pretrained_image_size),
            )
        
    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)
        self.apply(_init)
        nn.init.constant_(self.fc.weight, 0)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.normal_(self.positional_embedding.pos_embedding, std=0.02)  # _trunc_normal(self.positional_embedding.pos_embedding, std=0.02)
        nn.init.constant_(self.class_token, 0)

    def forward(self, x):
        """Breaks image into patches, applies transformer, applies MLP head.
        print(data.shape)
         Args:
            x (tensor): `b,c,fh,fw`
        """
        b, c, fh, fw = x.shape
        if self.hybrid == False:
            # Patch embedding based ImageTransformer
            x = DDFA.generate_uv_tex(x, self.patch_embedding, self.sequence_embedding, self.linear_1, self.proj, self.seq_embed, self.hybrid, self.device)
        else:
            # Hybrid ImageTransformer
            x = DDFA.generate_uv_tex(x, self.xception_feature_extractor, self.sequence_embedding, self.linear_1, self.proj, self.seq_embed, self.hybrid, self.device)
        
        if hasattr(self, 'class_token'):
            x = torch.cat((self.class_token.expand(b, -1, -1), x), dim=1)  # b,gh*gw+1,d
        if hasattr(self, 'positional_embedding'): 
            x = self.positional_embedding(x)  # b,gh*gw+1,d 

        x = self.transformer(x)  # b,gh*gw+1,d
        if hasattr(self, 'pre_logits'):
            x = self.pre_logits(x)
            x = torch.tanh(x)
        if hasattr(self, 'fc'):
            x = self.norm(x)[:, 0]  # b,d
            x = self.fc(x)  # b,num_classes
        return x