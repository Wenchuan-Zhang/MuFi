
import torch
import numpy as np 
# from x_transformers import CrossAttender

import torch
import torch.nn as nn
from torch import nn
from einops import reduce

# from x_transformers import Encoder
from torch.nn import ReLU

from models.layers.cross_attention import FeedForward, MMAttentionLayer, MMAttentionLayer2, MMAttentionLayer3
import pdb

import math
import pandas as pd
import torch.nn.functional as F

def exists(val):
    return val is not None


def SNN_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    import torch.nn as nn

    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            nn.AlphaDropout(p=dropout, inplace=False))

class SurvPath(nn.Module):
    def __init__(
        self, 
        omic_sizes=[100, 200, 300, 400, 500, 600],
        # wsi_embedding_dim=1024,
        wsi_embedding_dim=1536, #HIPT:384; Uni:1024; Virchow:2560; H-optimus:1536; Ctranspath:768; ViTS16:384; Conch:512; Kaiko:1024; Resnet:1024; CCL:2048;Phikon:768; 
        dropout=0.1,
        num_classes=4,
        wsi_projection_dim=256,
        omic_names = [],
        ):
        super(SurvPath, self).__init__()

        #---> general props
        self.num_pathways = len(omic_sizes)
        self.dropout = dropout

        #---> omics preprocessing for captum
        if omic_names != []:
            self.omic_names = omic_names
            all_gene_names = []
            for group in omic_names:
                all_gene_names.append(group)
            all_gene_names = np.asarray(all_gene_names)
            all_gene_names = np.concatenate(all_gene_names)
            all_gene_names = np.unique(all_gene_names)
            all_gene_names = list(all_gene_names)
            self.all_gene_names = all_gene_names

        #---> wsi props
        self.wsi_embedding_dim = wsi_embedding_dim #384
        self.wsi_projection_dim = wsi_projection_dim #256

        self.wsi_projection_net = nn.Sequential(
            nn.Linear(self.wsi_embedding_dim, self.wsi_projection_dim),
        )
        self.wsi_projection_net2 = nn.Sequential(
            nn.Linear(192, self.wsi_projection_dim // 2),
        )
        self.wsi_projection_net3 = nn.Sequential(
            nn.Linear(768, self.wsi_projection_dim // 4),
        )
        self.wsi_projection_net4 = nn.Sequential(
            nn.Linear(128, self.wsi_projection_dim // 8),
        )
        #---> omics props
        self.init_per_path_model(omic_sizes)

        #---> cross attention props
        self.identity = nn.Identity() # use this layer to calculate ig
        self.cross_attender = MMAttentionLayer(
            dim=self.wsi_projection_dim,
            dim_head=self.wsi_projection_dim // 2,
            heads=1,
            residual=False,
            dropout=0.1,
            num_pathways = self.num_pathways
        )
        self.cross_attender2 = MMAttentionLayer2(
            dim=self.wsi_projection_dim // 2,
            dim_head=self.wsi_projection_dim // 4,
            heads=1,
            residual=False,
            dropout=0.1,
            num_pathways = self.num_pathways
        )
        self.cross_attender3 = MMAttentionLayer3(
            dim=self.wsi_projection_dim // 4,
            dim_head=self.wsi_projection_dim // 8,
            heads=1,
            residual=False,
            dropout=0.1,
            num_pathways = self.num_pathways
        )
        #---> logits props 
        self.num_classes = num_classes
        self.feed_forward = FeedForward(self.wsi_projection_dim // 2, dropout=dropout)
        self.layer_norm = nn.LayerNorm(self.wsi_projection_dim // 2)
        self.feed_forward2 = FeedForward(self.wsi_projection_dim // 4, dropout=dropout)
        self.layer_norm2 = nn.LayerNorm(self.wsi_projection_dim // 4)
        self.feed_forward3 = FeedForward(self.wsi_projection_dim // 8, dropout=dropout)
        self.layer_norm3 = nn.LayerNorm(self.wsi_projection_dim // 8)
        # when both top and bottom blocks 
        self.to_logits = nn.Sequential(
                nn.Linear(256, int(self.wsi_projection_dim/4)),
                nn.ReLU(),
                nn.Linear(int(self.wsi_projection_dim/4), self.num_classes)
            )
        self.to_logits2 = nn.Sequential(
                nn.Linear(128, int(self.wsi_projection_dim/4)),
                nn.ReLU(),
                nn.Linear(int(self.wsi_projection_dim/4), self.num_classes)
            )
        
    def init_per_path_model(self, omic_sizes):
        hidden = [256, 256]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks)    
    
    def forward(self, **kwargs):

        wsi = kwargs['x_path1']
        num_wsi1patch = wsi.shape[1]
        num_wsi1pathway = self.num_pathways + num_wsi1patch
        # num_wsi1pathway = self.num_pathways
        wsi2 = kwargs['x_path2']
        num_wsi2patch = wsi2.shape[1]
        num_wsi2pathway = num_wsi1pathway + num_wsi2patch
        wsi3 = kwargs['x_path3']
        x_omic = [kwargs['x_omic%d' % i] for i in range(1,self.num_pathways+1)]
        mask = None
        return_attn = kwargs["return_attn"]
        
        #---> get pathway embeddings 
        h_omic = [self.sig_networks[idx].forward(sig_feat.float()) for idx, sig_feat in enumerate(x_omic)] ### each omic signature goes through it's own FC layer
        h_omic_bag = torch.stack(h_omic).unsqueeze(0) ### omic embeddings are stacked (to be used in co-attention)

        #---> project wsi to smaller dimension (same as pathway dimension)
        wsi_embed = self.wsi_projection_net(wsi) #1, n,1024> 1, n,256

        tokens = torch.cat([h_omic_bag, wsi_embed], dim=1) #1, 202+n, 256
        tokens = self.identity(tokens)
        
        if return_attn:
            mm_embed, attn_pathways, cross_attn_pathways, cross_attn_histology = self.cross_attender(x=tokens, mask=mask if mask is not None else None, return_attention=True)
        else:
            mm_embed = self.cross_attender(x=tokens, mask=mask if mask is not None else None, return_attention=False)

        #---> feedforward and layer norm 
        mm_embed = self.feed_forward(mm_embed)
        mm_embed = self.layer_norm(mm_embed) #1, 4427, 128
        paths_postSA_embed1 = mm_embed[:, :self.num_pathways, :]
        paths_postSA_embed1 = torch.mean(paths_postSA_embed1, dim=1) #1,128
        wsi_postSA_embed1 = mm_embed[:, self.num_pathways:, :] #1, n,128
        wsi_postSA_embed1 = torch.mean(wsi_postSA_embed1, dim=1)
        # mm_embed = mm_embed[:,:self.num_pathways,:] #1, 202, 128
        embedding1 = torch.cat([paths_postSA_embed1, wsi_postSA_embed1], dim=1) #---> both branches
        logits = self.to_logits(embedding1) #32*3>96

        # print(kwargs['case_id'])
        # print(wsi2.dtype)
        wsi_embed2 = self.wsi_projection_net2(wsi2) #add, 1, n2, 192 > 1, n2, 128
        tokens2 = torch.cat([mm_embed, wsi_embed2], dim=1) #1, 202+n2, 128
        tokens2 = self.identity(tokens2)
        if return_attn:
            mm_embed2, attn_pathways2, cross_attn_pathways2, cross_attn_histology2 = self.cross_attender2(x=tokens2, mask=mask if mask is not None else None, return_attention=True, num_wsi1pathway=num_wsi1pathway)
        else:
            mm_embed2 = self.cross_attender2(x=tokens2, mask=mask if mask is not None else None, return_attention=False, num_wsi1pathway=num_wsi1pathway)

        mm_embed2 = self.feed_forward2(mm_embed2)
        mm_embed2 = self.layer_norm2(mm_embed2) #1, 4427, 128

        wsi_embed3 = self.wsi_projection_net3(wsi3) #1, 768 > 1, 64
        tokens3 = torch.cat([mm_embed2, wsi_embed3], dim=1)
        tokens3 = self.identity(tokens3)
        if return_attn:
            mm_embed3, attn_pathways3, cross_attn_pathways3, cross_attn_histology3 = self.cross_attender3(x=tokens3, mask=mask if mask is not None else None, return_attention=True)
        else:
            mm_embed3 = self.cross_attender3(x=tokens3, mask=mask if mask is not None else None, return_attention=False)

        mm_embed3 = self.feed_forward3(mm_embed3)
        mm_embed3 = self.layer_norm3(mm_embed3) #1, 4427, 128
        #---> aggregate 
        # modality specific mean 
        paths_postSA_embed = mm_embed3[:, :self.num_pathways, :]
        paths_postSA_embed = torch.mean(paths_postSA_embed, dim=1) #1,128

        wsi_postSA_embed = mm_embed3[:, self.num_pathways:num_wsi1pathway, :]
        # wsi_postSA_embed = self.wsi_projection_net4(wsi_postSA_embed) #128 > 32
        wsi_postSA_embed = torch.mean(wsi_postSA_embed, dim=1)

        wsi_postSA_embed2 = mm_embed3[:, num_wsi1pathway:num_wsi2pathway, :]
        wsi_postSA_embed2 = torch.mean(wsi_postSA_embed2, dim=1)

        wsi_postSA_embed3 = mm_embed3[:, -1: , :]
        wsi_postSA_embed3 = torch.mean(wsi_postSA_embed3, dim=1)

        # when both top and bottom block
        # embedding = torch.cat([paths_postSA_embed, wsi_postSA_embed], dim=1) #---> both branches
        embedding = torch.cat([paths_postSA_embed, wsi_postSA_embed, wsi_postSA_embed2, wsi_postSA_embed3], dim=1) #---> both branches
        # embedding = paths_postSA_embed #---> top bloc only
        # embedding = wsi_postSA_embed #---> bottom bloc only

        # embedding = torch.mean(mm_embed, dim=1)
        #---> get logits
        logits2 = self.to_logits2(embedding) #32*3>96
        Y_prob = F.softmax(logits, dim = 1)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        if return_attn:
            return logits, attn_pathways, cross_attn_pathways, cross_attn_histology
        else:
            return logits, Y_hat, Y_prob, logits2
        
    def captum(self, omics_0 ,omics_1 ,omics_2 ,omics_3 ,omics_4 ,omics_5 ,omics_6 ,omics_7 ,omics_8 ,omics_9 ,omics_10 ,omics_11 ,omics_12 ,omics_13 ,omics_14 ,omics_15 ,omics_16 ,omics_17 ,omics_18 ,omics_19 ,omics_20 ,omics_21 ,omics_22 ,omics_23 ,omics_24 ,omics_25 ,omics_26 ,omics_27 ,omics_28 ,omics_29 ,omics_30 ,omics_31 ,omics_32 ,omics_33 ,omics_34 ,omics_35 ,omics_36 ,omics_37 ,omics_38 ,omics_39 ,omics_40 ,omics_41 ,omics_42 ,omics_43 ,omics_44 ,omics_45 ,omics_46 ,omics_47 ,omics_48 ,omics_49 ,omics_50 ,omics_51 ,omics_52 ,omics_53 ,omics_54 ,omics_55 ,omics_56 ,omics_57 ,omics_58 ,omics_59 ,omics_60 ,omics_61 ,omics_62 ,omics_63 ,omics_64 ,omics_65 ,omics_66 ,omics_67 ,omics_68 ,omics_69 ,omics_70 ,omics_71 ,omics_72 ,omics_73 ,omics_74 ,omics_75 ,omics_76 ,omics_77 ,omics_78 ,omics_79 ,omics_80 ,omics_81 ,omics_82 ,omics_83 ,omics_84 ,omics_85 ,omics_86 ,omics_87 ,omics_88 ,omics_89 ,omics_90 ,omics_91 ,omics_92 ,omics_93 ,omics_94 ,omics_95 ,omics_96 ,omics_97 ,omics_98 ,omics_99 ,omics_100 ,omics_101 ,omics_102 ,omics_103 ,omics_104 ,omics_105 ,omics_106 ,omics_107 ,omics_108 ,omics_109 ,omics_110 ,omics_111 ,omics_112 ,omics_113 ,omics_114 ,omics_115 ,omics_116 ,omics_117 ,omics_118 ,omics_119 ,omics_120 ,omics_121 ,omics_122 ,omics_123 ,omics_124 ,omics_125 ,omics_126 ,omics_127 ,omics_128 ,omics_129 ,omics_130 ,omics_131 ,omics_132 ,omics_133 ,omics_134 ,omics_135 ,omics_136 ,omics_137 ,omics_138 ,omics_139 ,omics_140 ,omics_141 ,omics_142 ,omics_143 ,omics_144 ,omics_145 ,omics_146 ,omics_147 ,omics_148 ,omics_149 ,omics_150 ,omics_151 ,omics_152 ,omics_153 ,omics_154 ,omics_155 ,omics_156 ,omics_157 ,omics_158 ,omics_159 ,omics_160 ,omics_161 ,omics_162 ,omics_163 ,omics_164 ,omics_165 ,omics_166 ,omics_167 ,omics_168 ,omics_169 ,omics_170 ,omics_171 ,omics_172 ,omics_173 ,omics_174 ,omics_175 ,omics_176 ,omics_177 ,omics_178 ,omics_179 ,omics_180 ,omics_181 ,omics_182 ,omics_183 ,omics_184 ,omics_185 ,omics_186 ,omics_187 ,omics_188 ,omics_189 ,omics_190 ,omics_191 ,omics_192 ,omics_193 ,omics_194 ,omics_195 ,omics_196 ,omics_197 ,omics_198 ,omics_199 ,omics_200 ,omics_201 ,omics_202 ,omics_203 ,omics_204 ,omics_205 ,omics_206 ,omics_207 ,omics_208 ,omics_209 ,omics_210 ,omics_211 ,omics_212 ,omics_213 ,omics_214 ,omics_215 ,omics_216 ,omics_217 ,omics_218 ,omics_219 ,omics_220 ,omics_221 ,omics_222 ,omics_223 ,omics_224 ,omics_225 ,omics_226 ,omics_227 ,omics_228 ,omics_229 ,omics_230 ,omics_231 ,omics_232 ,omics_233 ,omics_234 ,omics_235 ,omics_236 ,omics_237 ,omics_238 ,omics_239 ,omics_240 ,omics_241 ,omics_242 ,omics_243 ,omics_244 ,omics_245 ,omics_246 ,omics_247 ,omics_248 ,omics_249 ,omics_250 ,omics_251 ,omics_252 ,omics_253 ,omics_254 ,omics_255 ,omics_256 ,omics_257 ,omics_258 ,omics_259 ,omics_260 ,omics_261 ,omics_262 ,omics_263 ,omics_264 ,omics_265 ,omics_266 ,omics_267 ,omics_268 ,omics_269 ,omics_270 ,omics_271 ,omics_272 ,omics_273 ,omics_274 ,omics_275 ,omics_276 ,omics_277 ,omics_278 ,omics_279 ,omics_280 ,omics_281 ,omics_282 ,omics_283 ,omics_284 ,omics_285 ,omics_286 ,omics_287 ,omics_288 ,omics_289 ,omics_290 ,omics_291 ,omics_292 ,omics_293 ,omics_294 ,omics_295 ,omics_296 ,omics_297 ,omics_298 ,omics_299 ,omics_300 ,omics_301 ,omics_302 ,omics_303 ,omics_304 ,omics_305 ,omics_306 ,omics_307 ,omics_308 ,omics_309 ,omics_310 ,omics_311 ,omics_312 ,omics_313 ,omics_314 ,omics_315 ,omics_316 ,omics_317 ,omics_318 ,omics_319 ,omics_320 ,omics_321 ,omics_322 ,omics_323 ,omics_324 ,omics_325 ,omics_326 ,omics_327 ,omics_328 ,omics_329 ,omics_330, wsi):
        
        #---> unpack inputs
        mask = None
        return_attn = False
        
        omic_list = [omics_0 ,omics_1 ,omics_2 ,omics_3 ,omics_4 ,omics_5 ,omics_6 ,omics_7 ,omics_8 ,omics_9 ,omics_10 ,omics_11 ,omics_12 ,omics_13 ,omics_14 ,omics_15 ,omics_16 ,omics_17 ,omics_18 ,omics_19 ,omics_20 ,omics_21 ,omics_22 ,omics_23 ,omics_24 ,omics_25 ,omics_26 ,omics_27 ,omics_28 ,omics_29 ,omics_30 ,omics_31 ,omics_32 ,omics_33 ,omics_34 ,omics_35 ,omics_36 ,omics_37 ,omics_38 ,omics_39 ,omics_40 ,omics_41 ,omics_42 ,omics_43 ,omics_44 ,omics_45 ,omics_46 ,omics_47 ,omics_48 ,omics_49 ,omics_50 ,omics_51 ,omics_52 ,omics_53 ,omics_54 ,omics_55 ,omics_56 ,omics_57 ,omics_58 ,omics_59 ,omics_60 ,omics_61 ,omics_62 ,omics_63 ,omics_64 ,omics_65 ,omics_66 ,omics_67 ,omics_68 ,omics_69 ,omics_70 ,omics_71 ,omics_72 ,omics_73 ,omics_74 ,omics_75 ,omics_76 ,omics_77 ,omics_78 ,omics_79 ,omics_80 ,omics_81 ,omics_82 ,omics_83 ,omics_84 ,omics_85 ,omics_86 ,omics_87 ,omics_88 ,omics_89 ,omics_90 ,omics_91 ,omics_92 ,omics_93 ,omics_94 ,omics_95 ,omics_96 ,omics_97 ,omics_98 ,omics_99 ,omics_100 ,omics_101 ,omics_102 ,omics_103 ,omics_104 ,omics_105 ,omics_106 ,omics_107 ,omics_108 ,omics_109 ,omics_110 ,omics_111 ,omics_112 ,omics_113 ,omics_114 ,omics_115 ,omics_116 ,omics_117 ,omics_118 ,omics_119 ,omics_120 ,omics_121 ,omics_122 ,omics_123 ,omics_124 ,omics_125 ,omics_126 ,omics_127 ,omics_128 ,omics_129 ,omics_130 ,omics_131 ,omics_132 ,omics_133 ,omics_134 ,omics_135 ,omics_136 ,omics_137 ,omics_138 ,omics_139 ,omics_140 ,omics_141 ,omics_142 ,omics_143 ,omics_144 ,omics_145 ,omics_146 ,omics_147 ,omics_148 ,omics_149 ,omics_150 ,omics_151 ,omics_152 ,omics_153 ,omics_154 ,omics_155 ,omics_156 ,omics_157 ,omics_158 ,omics_159 ,omics_160 ,omics_161 ,omics_162 ,omics_163 ,omics_164 ,omics_165 ,omics_166 ,omics_167 ,omics_168 ,omics_169 ,omics_170 ,omics_171 ,omics_172 ,omics_173 ,omics_174 ,omics_175 ,omics_176 ,omics_177 ,omics_178 ,omics_179 ,omics_180 ,omics_181 ,omics_182 ,omics_183 ,omics_184 ,omics_185 ,omics_186 ,omics_187 ,omics_188 ,omics_189 ,omics_190 ,omics_191 ,omics_192 ,omics_193 ,omics_194 ,omics_195 ,omics_196 ,omics_197 ,omics_198 ,omics_199 ,omics_200 ,omics_201 ,omics_202 ,omics_203 ,omics_204 ,omics_205 ,omics_206 ,omics_207 ,omics_208 ,omics_209 ,omics_210 ,omics_211 ,omics_212 ,omics_213 ,omics_214 ,omics_215 ,omics_216 ,omics_217 ,omics_218 ,omics_219 ,omics_220 ,omics_221 ,omics_222 ,omics_223 ,omics_224 ,omics_225 ,omics_226 ,omics_227 ,omics_228 ,omics_229 ,omics_230 ,omics_231 ,omics_232 ,omics_233 ,omics_234 ,omics_235 ,omics_236 ,omics_237 ,omics_238 ,omics_239 ,omics_240 ,omics_241 ,omics_242 ,omics_243 ,omics_244 ,omics_245 ,omics_246 ,omics_247 ,omics_248 ,omics_249 ,omics_250 ,omics_251 ,omics_252 ,omics_253 ,omics_254 ,omics_255 ,omics_256 ,omics_257 ,omics_258 ,omics_259 ,omics_260 ,omics_261 ,omics_262 ,omics_263 ,omics_264 ,omics_265 ,omics_266 ,omics_267 ,omics_268 ,omics_269 ,omics_270 ,omics_271 ,omics_272 ,omics_273 ,omics_274 ,omics_275 ,omics_276 ,omics_277 ,omics_278 ,omics_279 ,omics_280 ,omics_281 ,omics_282 ,omics_283 ,omics_284 ,omics_285 ,omics_286 ,omics_287 ,omics_288 ,omics_289 ,omics_290 ,omics_291 ,omics_292 ,omics_293 ,omics_294 ,omics_295 ,omics_296 ,omics_297 ,omics_298 ,omics_299 ,omics_300 ,omics_301 ,omics_302 ,omics_303 ,omics_304 ,omics_305 ,omics_306 ,omics_307 ,omics_308 ,omics_309 ,omics_310 ,omics_311 ,omics_312 ,omics_313 ,omics_314 ,omics_315 ,omics_316 ,omics_317 ,omics_318 ,omics_319 ,omics_320 ,omics_321 ,omics_322 ,omics_323 ,omics_324 ,omics_325 ,omics_326 ,omics_327 ,omics_328 ,omics_329 ,omics_330]

        #---> get pathway embeddings 
        h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(omic_list)] ### each omic signature goes through it's own FC layer
        h_omic_bag = torch.stack(h_omic, dim=1)  #.unsqueeze(0) ### omic embeddings are stacked (to be used in co-attention)
        
        #---> project wsi to smaller dimension (same as pathway dimension)
        wsi_embed = self.wsi_projection_net(wsi)

        tokens = torch.cat([h_omic_bag, wsi_embed], dim=1)
        tokens = self.identity(tokens)

        if return_attn:
            mm_embed, attn_pathways, cross_attn_pathways, cross_attn_histology = self.cross_attender(x=tokens, mask=mask if mask is not None else None, return_attention=True)
        else:
            mm_embed = self.cross_attender(x=tokens, mask=mask if mask is not None else None, return_attention=False)

        #---> feedforward and layer norm 
        mm_embed = self.feed_forward(mm_embed)
        mm_embed = self.layer_norm(mm_embed)
        
        #---> aggregate 
        # modality specific mean 
        paths_postSA_embed = mm_embed[:, :self.num_pathways, :]
        paths_postSA_embed = torch.mean(paths_postSA_embed, dim=1)

        wsi_postSA_embed = mm_embed[:, self.num_pathways:, :]
        wsi_postSA_embed = torch.mean(wsi_postSA_embed, dim=1)
        embedding = torch.cat([paths_postSA_embed, wsi_postSA_embed], dim=1)

        #---> get logits
        logits = self.to_logits(embedding)

        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(survival, dim=1)

        if return_attn:
            return risk, attn_pathways, cross_attn_pathways, cross_attn_histology
        else:
            return risk