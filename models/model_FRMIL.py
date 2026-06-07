import torch
import torch.nn as nn
import torch.nn.functional as F
import math, numpy as np
from models.model_utils import *
class MAB(nn.Module):
    def __init__(self, dim_Q, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V     = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_Q, dim_V)
        self.fc_v = nn.Linear(dim_Q, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, inst_mode=False):
        Q    = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        
        if inst_mode:
            return O
        else:
            return O.squeeze(1)


class FRMIL(nn.Module):
    def __init__(self, omic_input_dim=None, fusion=None, size_arg = "small", dropout=0.25, n_classes=4, df_comp=None, dim_per_path_1=16, dim_per_path_2=64, device="cpu"):
        super(FRMIL,self).__init__()
        self.device = device
        # dim_hidden       = 1536
        num_heads        = 8
        self.k           = 1
        self.fusion = fusion #add
        self.size_dict_path = {"small": [1536, 256, 256], "big": [1024, 512, 384]}
        size = self.size_dict_path[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        self.fc = nn.Sequential(*fc)
        dim_hidden       = size[1]
        # Encoder for omic/genomic data
        self.enc = nn.Sequential(
            nn.Linear(size[1], 1),
            nn.Sigmoid()
        )
        
        self.cls_token = nn.Parameter(torch.Tensor(1, 1, dim_hidden))
        nn.init.xavier_uniform_(self.cls_token)
        
        self.conv_head = torch.nn.Conv2d(dim_hidden, dim_hidden, 3, 1, 3//2, groups=dim_hidden)
        torch.nn.init.xavier_uniform_(self.conv_head.weight)
        
        self.selt_att = MAB(size[1], size[1], num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(size[1], n_classes),
        )
        self.df_comp = df_comp
        self.dim_per_path_1 = dim_per_path_1
        self.num_pathways = self.df_comp.shape[1]
        self.dim_per_path_2 = dim_per_path_2
        self.input_dim = omic_input_dim
        # Feature fusion module  #add
        ### Constructing Genomic SNN
        if self.fusion is not None:
            self.num_pathways = self.df_comp.shape[1]
            M_raw = torch.Tensor(self.df_comp.values)
            self.mask_1 = torch.repeat_interleave(M_raw, self.dim_per_path_1, dim=1)

            self.fc_1_weight = nn.init.xavier_normal_(nn.Parameter(torch.FloatTensor(self.input_dim, self.dim_per_path_1*self.num_pathways)))
            self.fc_1_bias = nn.Parameter(torch.rand(self.dim_per_path_1*self.num_pathways))

            self.fc_2_weight = nn.init.xavier_normal_(nn.Parameter(torch.FloatTensor(self.dim_per_path_1*self.num_pathways, self.dim_per_path_2*self.num_pathways)))
            self.fc_2_bias = nn.Parameter(torch.rand(self.dim_per_path_2*self.num_pathways))

            self.mask_2 = np.zeros([self.dim_per_path_1*self.num_pathways, self.dim_per_path_2*self.num_pathways])
            for (row, col) in zip(range(0, self.dim_per_path_1*self.num_pathways, self.dim_per_path_1), range(0, self.dim_per_path_2*self.num_pathways, self.dim_per_path_2)):
                self.mask_2[row:row+self.dim_per_path_1, col:col+self.dim_per_path_2] = 1
            self.mask_2 = torch.Tensor(self.mask_2)

            self.upscale = nn.Sequential(
                nn.Linear(self.dim_per_path_2*self.num_pathways, int(256/4)),
                nn.ReLU(),
                nn.Linear(int(256/4), 256)
            )

            if fusion == "concat":
                self.mm = nn.Sequential(*[nn.Linear(256*2, size[2]), nn.ReLU(), nn.Linear(size[2], size[2]), nn.ReLU()])
            elif fusion == 'bilinear':
                self.mm = BilinearFusion(dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)
            else:
                self.mm = None

            self.fc_1_weight.to(self.device)
            self.fc_1_bias.to(self.device)
            self.mask_1 = self.mask_1.to(self.device)
            self.fc_2_weight.to(self.device)
            self.fc_2_bias.to(self.device)
            self.mask_2 = self.mask_2.to(self.device)
            self.mm = self.mm.to(self.device)
        self.activation = nn.ReLU()

    def recalib(self, inputs, option='max'):
        A1, Q = [], []
        bs = inputs.shape[0]
        if option == 'mean':
            Q = torch.mean(inputs, dim=1, keepdim=True)
            A1 = self.enc(Q.squeeze(1))
            return A1, Q
        else:
            for i in range(bs):
                a1 = self.enc(inputs[i].unsqueeze(0)).squeeze(0)
                _, m_indices = torch.sort(a1, 0, descending=True)
                feat_q = []
                len_i = m_indices.shape[0] - 1
                for i_q in range(self.k):
                    if option == 'max':
                        feats = torch.index_select(inputs[i], dim=0, index=m_indices[i_q, :])
                    else:
                        feats = torch.index_select(inputs[i], dim=0, index=m_indices[len_i - i_q, :])
                    feat_q.append(feats)
                    
                feats = torch.stack(feat_q)
                A1.append(a1.squeeze(1))
                Q.append(feats.mean(0))
            A1 = torch.stack(A1)
            Q = torch.stack(Q)
            return A1, Q
            
    def forward(self, **kwargs):
        inputs = kwargs['data_WSI']
        inputs = self.fc(inputs)
        A1, Q = self.recalib(inputs, 'max')
        
        ##################################################################
        inputs  = F.relu(inputs - Q)
        i_shift = inputs
        ##################################################################

        ##################################################################
        # ----> Pad inputs 
        H = inputs.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        inputs = torch.cat([inputs, inputs[:,:add_length,:]],dim=1)
        
        B = inputs.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        inputs = torch.cat((cls_tokens, inputs), dim=1)
        
        B, _, C = inputs.shape
        cls_token, feat_token = inputs[:, 0], inputs[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, _H, _W)
        cnn_feat = self.conv_head(cnn_feat) + cnn_feat
        x = cnn_feat.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        ##################################################################

        ##################################################################
        # Bag pooling with critical feature
        h_path = self.selt_att(Q, x).squeeze()
        
        # Fusion with omic data #add
        if self.fusion is not None:
            x_omic = kwargs['data_omics']
            x_omic = x_omic.squeeze()

            out = torch.matmul(x_omic, self.fc_1_weight * self.mask_1) + self.fc_1_bias
            out = self.activation(out)
            out = torch.matmul(out, self.fc_2_weight * self.mask_2) + self.fc_2_bias 

            #---> apply linear transformation to upscale the dim_per_pathway (from 32 to 256) Lin, GELU, dropout, 
            h_omic = self.upscale(out)

            if self.fusion == 'concat':
                h = self.mm(torch.cat([h_path, h_omic], axis=0))
            elif self.fusion == 'bilinear':
                h = self.mm(h_path.unsqueeze(dim=0), h_omic.unsqueeze(dim=0)).squeeze()
        else:
            h = h_path # [256] vector

        ##################################################################
        logits  = self.classifier(h).unsqueeze(0) # logits needs to be a [1 x 4] vector 
        Y_prob = F.softmax(logits, dim = 1)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]

        return logits,Y_hat, Y_prob