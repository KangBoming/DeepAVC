import torch
from torch import nn

class ProjectionMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = out_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class CA_DTI_Layer(nn.Module):
    def __init__(self, 
                 d_model, 
                 nhead, 
                 dropout=0.1, 
                 activation=nn.ReLU(),
                 layer_norm_eps=1e-5, 
                 batch_first=False, 
                 norm_first=False):
        super(CA_DTI_Layer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

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

        self.activation = activation

    def forward(self, tgt, memory, 
                tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, 
                memory_key_padding_mask=None, 
                return_attn_weights=False):

        x = tgt

        sa_output, self_attn_weights = self._sa_block(x, tgt_mask, tgt_key_padding_mask)
        x = self.norm1(x + sa_output)

        mha_output, cross_attn_weights = self._mha_block(x, memory, memory_mask, memory_key_padding_mask)
        x = self.norm2(x + mha_output)

        x = self.norm3(x + self._ff_block(x))

        if return_attn_weights:
            return x, self_attn_weights, cross_attn_weights
        else:
            return x

    def _sa_block(self, x, attn_mask, key_padding_mask):
        attn_output, attn_weights = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True
        )
        return self.dropout1(attn_output), attn_weights

    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        attn_output, attn_weights = self.multihead_attn(
            x, mem, mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True
        )
        return self.dropout2(attn_output), attn_weights

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)



class CA_DTI_MLP(nn.Module):
    def __init__(self, 
                 d_model, 
                 n_heads, 
                 num_layers, 
                 kpgt_model,
                 smiles_dim=768, 
                 kpgt_features_dim=2304,
                 protein_dim=5120, 
                 mlp_hidden_dim=256, 
                 num_classes=1,
                 dropout=0.1, 
                 layer_norm_eps=1e-5, 
                 return_attn=False):
        
        super(CA_DTI_MLP, self).__init__()

        self.n_heads = n_heads
        self.return_attn = return_attn

        # === Cross-attention layers ===
        self.smiles_layers = nn.ModuleList([
            nn.ModuleDict({
                "layer": CA_DTI_Layer(
                    d_model, n_heads, dropout=dropout,
                    layer_norm_eps=layer_norm_eps,
                    batch_first=True
                )
            }) for _ in range(num_layers)
        ])

        self.protein_layers = nn.ModuleList([
            nn.ModuleDict({
                "layer": CA_DTI_Layer(
                    d_model, n_heads, dropout=dropout,
                    layer_norm_eps=layer_norm_eps,
                    batch_first=True
                )
            }) for _ in range(num_layers)
        ])

        # === KPGT frozen feature extractor ===
        self.kpgt_model = kpgt_model
        for param in self.kpgt_model.parameters():
            param.requires_grad = False


        self.smiles_fc = ProjectionMLP(
            in_dim=smiles_dim,
            out_dim=d_model,
            hidden_dim=d_model,
            dropout=dropout
        )

        self.protein_fc = ProjectionMLP(
            in_dim=protein_dim,
            out_dim=d_model,
            hidden_dim=d_model,
            dropout=dropout
        )

        self.kpgt_feature_fc = ProjectionMLP(
            in_dim=kpgt_features_dim,
            out_dim=d_model,
            hidden_dim=d_model,
            dropout=dropout
        )

        self.protein_raw_fc = ProjectionMLP(
            in_dim=protein_dim,
            out_dim=d_model,
            hidden_dim=d_model,
            dropout=dropout
        )

        # === Pooling ===
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # === MLP classifier (NOTE: input dim = 4 * d_model) ===
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 4, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, num_classes)
        )


    def forward(self, graphs, fps, mds, protein_features):
        """
        graphs: molecular graphs
        fps, mds: molecular fingerprints / descriptors
        protein_features: (B, L_pro, protein_dim)
        """

        # === KPGT feature extraction (frozen) ===
        with torch.no_grad():
            smiles_features, smiles_mask, g_feats = \
                self.kpgt_model.generate_node_feature(graphs, fps, mds)

        # === Protein padding mask (True = padding) ===
        protein_mask = (protein_features == 0).all(dim=-1)

        # === Linear projections ===
        x = self.smiles_fc(smiles_features)         # (B, N_atom, d_model)
        y = self.protein_fc(protein_features)       # (B, L_pro, d_model)
        z = self.kpgt_feature_fc(g_feats)            # (B, d_model)

        attn_weights = {}

        # === Bidirectional cross-attention ===
        for smi_layer, pro_layer in zip(self.smiles_layers, self.protein_layers):
            x, _, smi2pro_attn = smi_layer["layer"](
                x, y,
                tgt_key_padding_mask=smiles_mask,
                memory_key_padding_mask=protein_mask,
                return_attn_weights=self.return_attn
            )

            y, _, pro2smi_attn = pro_layer["layer"](
                y, x,
                tgt_key_padding_mask=protein_mask,
                memory_key_padding_mask=smiles_mask,
                return_attn_weights=self.return_attn
            )

            if self.return_attn:
                attn_weights["smi2pro_attn"] = smi2pro_attn
                attn_weights["pro2smi_attn"] = pro2smi_attn

        # === Pool cross-attended features ===
        x_pooled = self.avg_pool(x.transpose(1, 2)).squeeze(-1)  # (B, d_model)
        y_pooled = self.avg_pool(y.transpose(1, 2)).squeeze(-1)  # (B, d_model)

        # === NEW: pooled raw protein embedding (mask-aware) ===
        valid_mask = (~protein_mask).unsqueeze(-1).float()       # (B, L_pro, 1)
        protein_raw_mean = (
            (protein_features * valid_mask).sum(dim=1) /
            valid_mask.sum(dim=1).clamp(min=1.0)
        )                                                        # (B, protein_dim)
        protein_raw_proj = self.protein_raw_fc(protein_raw_mean)  # (B, d_model)

        # === Final concatenation ===
        concat_features = torch.cat(
            [x_pooled, y_pooled, z, protein_raw_proj],
            dim=1
        )  # (B, 4 * d_model)

        output = self.mlp(concat_features)

        if self.return_attn:
            return output, attn_weights
        else:
            return output








