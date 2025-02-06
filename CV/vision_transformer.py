import torch
import torch.nn as nn

from functools import partial # for partial function application
from collections import OrderedDict 

# Dropout 在神经元级别随机置零，DropPath 在样本/路径级别随机丢弃。
def drop_path(x,drop_prob:float=0.,training:bool=False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survial rate' as the argument.
    """
    #when drop_prob is 0 or not training, return original input
    if drop_prob==0. or not training:
        return x 
    keep_prob = 1 - drop_prob
    #generate a shape matching random tensor
    #eg. for a 4D tensor(B,C,H,W), we generate a tensor of shape (B,1,1,1)
    #the purpose is to generate random binary mask in batch dimension,other dimensions share the same shape
    shape = (x.shape[0],)+(1,)*(x.ndim-1)
    # torch.rand 生成 [0, 1) 的均匀分布随机数。
    # 将随机数加上 keep_prob，得到范围为 [keep_prob, 1 + keep_prob) 的张量。
    # 通过 floor_() 向下取整，将张量二值化为 0 或 1：
    # 若随机数 < 1，结果为 0（丢弃路径）。
    # 若随机数 ≥ 1，结果为 1（保留路径）。
    # 每个样本被保留的概率为 keep_prob，丢弃概率为 drop_prob。
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    # 缩放：将输入 x 除以 keep_prob，例如 keep_prob=0.8 时，缩放因子为 1/0.8 = 1.25。
    # 掩码：将缩放后的 x 与二值掩码 random_tensor 相乘，随机丢弃部分样本的路径。
    # 缩放操作是为了保持输出的期望值与原输入一致（类似 Dropout 的缩放机制）。
    # 掩码操作在训练时随机丢弃路径，增强模型鲁棒性。
    output = x.div(keep_prob)*random_tensor
    return output

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768,norm_layer=None):
        super().__init__()
        # basic settings
        img_size = (img_size,img_size)
        patch_size = (patch_size,patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0]//patch_size[0],img_size[1]//patch_size[1])  
        # projection layer  
        self.proj=nn.Conv2d(in_channels=in_channels,
                            out_channels=embed_dim,
                            kernel_size=patch_size,
                            stride=patch_size)
        self.norm=nn.LayerNorm(embed_dim) if norm_layer else nn.Identity()
    def forward(self,x):
        #判断输入图片符合尺寸要求
        B,C,H,W = x.shape
        assert H==self.img_size[0] and W==self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # project patches to embedding dim
        # flatten:[B,C,H,W] -> [B,C,HW]
        # transpose:[B,C,HW] -> [B,HW,C] 
        # 输入：假设 self.proj 输出形状为 [B, D, H_patch, W_patch]（例如 [B, 768, 14, 14]）。
        # 操作：flatten(2) 将第 3 个维度（H_patch）和第 4 个维度（W_patch）展平为一个维度。
        # 输出形状：[B, D, H_patch*W_patch]（例如 [B, 768, 14*14] = [B, 768, 196]）。
        # Transformer 的输入需要是一个序列，其形状为 [Batch, Sequence_Length, Feature_Dim]。通过 transpose，将：
        # Sequence_Length 设为 H_patch*W_patch（即 196），表示序列中有 196 个 Patch。
        # Feature_Dim 设为 D（即 768），表示每个 Patch 的特征向量维度。
        x=self.proj(x).flatten(2).transpose(1,2)
        x=self.norm(x)
        return x

class Attention(nn.Module):
    def __init__(self,
                 dim, # token demension
                 num_heads=8,
                 qkv_bias=False,# 
                 qkv_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads # single head dim
        self.scale = qkv_scale or head_dim ** -0.5 # scale factor

        #通过一个线性层将输入 dim 维的token映射为 3*dim 维，再拆分为Q、K、V三个张量
        # 输入：[B, N, D]（Batch, 序列长度, 嵌入维度）。
        # 输出：[B, N, 3*D] → 拆分为 [B, N, D] ×3（Q、K、V各占一份）
        self.qkv = nn.Linear(dim,dim*3,bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        # proj: 将多头注意力的输出拼接后，通过线性层融合信息，保持输出维度与输入一致
        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
    def forward(self, x):
        B , N, D = x.shape
        # 目标生成q,k,v[B,num_heads,N,D],那么qkv为[3,B,num_heads,N,D]
        # 根据这个硬拆分就行了
        qkv=self.qkv(x).reshape(B,N,3,self.num_heads,self.D//self.num_heads).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]

        attn=(q@k.transpose(-2,-1))*self.scale
        # 第一个N是query dim 第二个N是key dim 我们需要在key的维度上做softmax(我们要对一个query的每一个key做softmax)
        attn=attn.softmax(dim=-1)   
        attn=self.attn_drop(attn)   
        # 这里的attn是[B,num_heads,N,N]
        # 乘以v后是[B,num_heads,N,D]
        # 转置后是[B,N,num_heads,D]
        # 然后合并维度后是[B,N,D]
        x=(attn @ v).transpose(1,2).reshape(B,N,D)
        x=self.proj(x)
        x=self.proj_drop(x)
        return x

#MLP模块(easy to understand)
class MLP(nn.Module):
    def __init__(self,in_features,hidden_features=None,output_features=None,act_Layer=nn.GELU,drop=0.):
        super().__init__()
        hidden_features=hidden_features or in_features
        output_features=output_features or in_features
        self.fc1=nn.Linear(in_features,hidden_features)
        self.act=act_Layer()
        self.fc2=nn.Linear(hidden_features,output_features)
        self.drop=nn.Dropout(drop)
    
    def forward(self,x):
        x=self.fc1(x)
        x=self.act(x)
        x=self.drop(x)
        x=self.fc2(x)
        x=self.drop(x)
        return x
        
class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,# 这个表示最后一个全连接层输入是前面输入的4倍
                 qkv_bias=False,
                 qkv_scale=None,
                 drop_ratio=0,
                 attn_drop_ratio=0,
                 drop_path_ratio=0,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1=norm_layer(dim)
        self.attn=Attention(dim,num_heads,qkv_bias,qkv_scale,attn_drop_ratio,drop_ratio)
        self.drop_path=DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2=norm_layer(dim)
        mlp_hidden_dim=int(dim*mlp_ratio)
        self.mlp=MLP(in_features=dim,hidden_features=mlp_hidden_dim,act_Layer=act_layer,drop=drop_ratio)
    
    def forward(self,x):
        x=x+self.drop_path(self.attn(self.norm1(x)))
        x=x+self.drop_path(self.mlp(self.norm2(x)))
        return x
        
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer,self).__init__()
        self.num_classes=num_classes
        self.num_features=self.embed_dim=embed_dim # num_features for consistency with other models
        self.num_tokens=2 if distilled else 1
        norm_layer=norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer=act_layer or nn.GELU

        self.patch_embed=embed_layer(img_size=img_size,patch_size=patch_size,in_channels=in_channels,embed_dim=embed_dim)
        num_patches=self.patch_embed.num_patches
        # 第一个1是batch 后面是1*768 去cat 196*768
        self.cls_token=nn.Parameter(torch.zeros(1,1,embed_dim))
        self.dist_token=nn.Parameter(torch.zeros(1,1,embed_dim)) if distilled else None
        self.pos_embed=nn.Parameter(torch.zeros(1,num_patches+self.num_tokens,embed_dim))
        self.pos_drop=nn.Dropout(p=drop_ratio)
        
        #12个block里面我们采取drop path ratio递增
        dpr=[x.item() for x in torch.linspace(0,drop_path_ratio,depth)]
        # * 表示将list给展开迭代出来
        self.blocks=nn.Sequential(*[
            Block(dim=embed_dim,num_heads=num_heads,mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,qk_scale=qk_scale,
                  drop_ratio=drop_ratio,attn_drop_ratio=attn_drop_ratio,drop_path_ratio=dpr[i],
                  norm_layer=norm_layer,act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm=norm_layer(embed_dim)

        # Representation layer(这个层是用来做分类用的)
        if representation_size and not distilled:
            self.has_logits=True
            self.num_features=representation_size
            self.pre_logits=nn.Sequential(OrderedDict([
                ('fc',nn.Linear(embed_dim, representation_size),
                 ('act',nn.Tanh()))
            ]))
        else:
            self.has_logits=False
            self.pre_logits=nn.Identity()
        
        # Classifier head(分类头)
        self.head = nn.Linear(self.num_features,num_classes) if num_classes>0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
        
        # Weight initalization
        nn.init.trunc_normal_(self.pos_embed,std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token,std=0.02)
        nn.init.trunc_normal_(self.cls_token,std=0.02)
        self.apply(_init_vit_weights)
    
    def forward_features(self,x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token=self.cls_token.expand(x.shape[0],-1,-1)
        # 拼接
        if self.dist_token is None:
            x=torch.cat((x,cls_token),dim=1) #[B, 197, 768]
        else:
            x=torch.cat((x,self.dist_token.expand(x.shape[0],-1,-1)),dim=1)
        # 位置编码
        x=x+self.pos_embed
        x=self.pos_drop(x)
        # transformer
        x=self.blocks(x)
        x=self.norm(x)
        if self.dist_token is None:
            # [B, 197, 768] -> [B, 768]
            # 提取每一个batch中的第一个token（cls token）
            return self.pre_logits(x[:,0])
        else:
            return x[:,0], x[:,1]
    
    def forward(self,x):
        x=self.forward_features(x)
        # 蒸馏的情况下
        if self.head_dist is not None:
            x,x_dist=self.head(x[0]),self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                return x,x_dist
            #during inference, return the average of both classifier predictions    
            else:
                return (x+x_dist)/2
        else:
            x=self.head(x)
        return x
    
def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base_patch16_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224(num_classes: int = 1000):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_huge_patch14_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: converted weights not currently available, too large for github release hosting.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=14,
                              embed_dim=1280,
                              depth=32,
                              num_heads=16,
                              representation_size=1280 if has_logits else None,
                              num_classes=num_classes)
    return model    