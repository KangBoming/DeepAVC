import torch
from torch import nn


class CADTI_Layer(nn.Module):
    def __init__(self, 
                 d_model, 
                 nhead, 
                 dropout=0.1, 
                 activation=nn.ReLU(),
                 layer_norm_eps=1e-5, 
                 batch_first=False, 
                 norm_first=False):
        super(CADTI_Layer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        # Implementation of Feedforward model
        self.dim_feedforward = int(4 * d_model)
        self.linear1 = nn.Linear(d_model, self.dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(self.dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Activation function
        self.activation = activation

    def forward(self, tgt, memory, 
                tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, 
                memory_key_padding_mask=None, 
                return_attn_weights=False):
        
        x = tgt

        if self.norm_first:
            sa_output, self_attn_weights = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + sa_output
            mha_output, cross_attn_weights = self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + mha_output
            x = x + self._ff_block(self.norm3(x))
        else:
            sa_output, self_attn_weights = self._sa_block(x, tgt_mask, tgt_key_padding_mask)
            x = self.norm1(x + sa_output)
            mha_output, cross_attn_weights = self._mha_block(x, memory, memory_mask, memory_key_padding_mask)
            x = self.norm2(x + mha_output)
            x = self.norm3(x + self._ff_block(x))

        if return_attn_weights:
            return x, self_attn_weights, cross_attn_weights
        else:
            return x

    # self-attention block
    def _sa_block(self, x, attn_mask, key_padding_mask):


        attn_output, attn_weights = self.self_attn(
            x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=True
        )
        return self.dropout1(attn_output), attn_weights

    # multihead attention block
    def _mha_block(self, x, mem, attn_mask, key_padding_mask):

        attn_output, attn_weights = self.multihead_attn(
            x, mem, mem, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=True
        )
        return self.dropout2(attn_output), attn_weights

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)



class CADTI_Finetune(nn.Module):
    def __init__(self, 
                 d_model, 
                 n_heads, 
                 num_layers, 
                 kpgt_model,
                 smiles_dim=768, 
                 kpgt_features_dim = 2304,
                 protein_dim=1280, 

                 mlp_hidden_dim=256, 
                 num_classes=1,
                 dropout=0.1, 
                 layer_norm_eps=1e-5, 
                 return_attn=False):
        
        super(CADTI_Finetune, self).__init__()

        self.n_heads = n_heads

        self.smiles_layers = nn.ModuleList([
            nn.ModuleDict({
               
                'layer': CADTI_Layer(d_model, n_heads, dropout=dropout,
                                                         layer_norm_eps=layer_norm_eps, batch_first=True, norm_first=False),
                'layer_norm': nn.LayerNorm(d_model, eps=layer_norm_eps)
            })
            for _ in range(num_layers)
        ])


        self.protein_layers = nn.ModuleList([
            nn.ModuleDict({
                
                'layer': CADTI_Layer(d_model, n_heads, dropout=dropout,
                                    layer_norm_eps=layer_norm_eps, batch_first=True, norm_first=False),
                'layer_norm': nn.LayerNorm(d_model, eps=layer_norm_eps)
            })
            for _ in range(num_layers)

        ])

        self.return_attn = return_attn

        # 使用KPGT模型提取node feature
        self.kpgt_model = kpgt_model

        # # 冻结 kpgt_model 的所有参数(可选)
        # for param in self.kpgt_model.parameters():
        #     param.requires_grad = False

        # 将药物和蛋白质特征映射到相同大小的维度
        self.smiles_fc = nn.Linear(smiles_dim, d_model) # 768 ->256
        self.protein_fc = nn.Linear(protein_dim, d_model) # 1280 -> 256
        self.kpgt_feature_fc = nn.Linear(kpgt_features_dim, d_model) # 2304 -> 256

        # 平均池化
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # MLP 分类器
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 3, mlp_hidden_dim), # 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, num_classes)
        )

    def forward(self, graphs, fps, mds, protein_features):

        ### 生成药物graph的节点特征和带有fps和mds的分子指纹特征
        smiles_features, smiles_mask,g_feats = self.kpgt_model.generate_node_feature(graphs, fps, mds)


        protein_mask = (protein_features == 0).all(dim=-1)

        x = self.smiles_fc(smiles_features)
        y = self.protein_fc(protein_features)

        z = self.kpgt_feature_fc(g_feats)  # (batch_size, d_model)

        attn_weights = {}

        for smiles_layer, protein_layer in zip(self.smiles_layers, self.protein_layers):

            x,  _, cross_attn_smiles= smiles_layer['layer'](x, y, tgt_key_padding_mask=smiles_mask, # query
                                                               memory_key_padding_mask=protein_mask, # key
                                                               return_attn_weights=self.return_attn)

            y, _, cross_attn_protein = protein_layer['layer'](y, x, tgt_key_padding_mask=protein_mask, # query
                                                                memory_key_padding_mask=smiles_mask, # key
                                                                return_attn_weights=self.return_attn)
            if self.return_attn:
                attn_weights['smi2pro_attn'] = cross_attn_smiles
                attn_weights['pro2smi_attn'] = cross_attn_protein

        x_pooled = self.avg_pool(x.transpose(1, 2)).squeeze(-1)  # (batch_size, d_model)
        y_pooled = self.avg_pool(y.transpose(1, 2)).squeeze(-1)  # (batch_size, d_model  )
        concat_features = torch.cat([x_pooled, y_pooled, z], dim=1)  # (batch_size, d_model * 3)
        output = self.mlp(concat_features)

        if self.return_attn:
            return output, attn_weights
        else:
            return output














