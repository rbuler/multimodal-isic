import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

class AttentionFusion(nn.Module):
    def __init__(self, input_dim, num_modalities):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, features):
        stacked = torch.stack(features, dim=1)  # [B, M, D]
        B, M, D = stacked.size()
        stacked_flat = stacked.view(B * M, D)
        scores_flat = self.attn(stacked_flat)  # [B*M, 1]
        scores = scores_flat.view(B, M)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)  # [B, M, 1]
        fused = (stacked * weights).sum(dim=1)  # [B, D]
        return fused

class AttentionFusion_Late(nn.Module):
    def __init__(self, num_modalities, num_classes=7):
        super().__init__()
        self.attention_net = nn.Sequential(
        nn.Linear(num_modalities * num_classes, 128),
        nn.ReLU(),
        nn.Linear(128, num_modalities)
    )

    def forward(self, logits):
            concat_logits = torch.cat(logits, dim=1)  # [batch_size, num_modalities * num_classes]
            attn_scores = self.attention_net(concat_logits)  # [batch_size, num_modalities]
            attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(2)  # [batch_size, num_modalities, 1]
            weighted_logits = torch.stack(logits, dim=1) * attn_weights
            fused_logits = weighted_logits.sum(dim=1)  # [batch_size, num_classes]
            return fused_logits

class MultiModalFusionNet(nn.Module):
    def __init__(self,
                 modality=['image', 'radiomics', 'clinical', 'artifacts'],
                 fusion_level='intermediate',  # 'intermediate' or 'late'
                 fusion_strategy='attention',  # 'concat', 'weighted', 'attention'
                 radiomics_dim=102,
                 num_sex_classes=3,
                 num_loc_classes=15,
                 num_artifact_classes=6,
                 num_classes=7):

        super().__init__()
        self.modality = modality
        self.fusion_level = fusion_level
        self.fusion_strategy = fusion_strategy

        self.image_model = EfficientNet.from_pretrained('efficientnet-b3')
        self.image_model._fc = nn.Identity()
        self.image_proj = nn.Sequential(
            nn.Linear(1536, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2)
        )

        self.radiomics_mlp = nn.Sequential(
            nn.Linear(radiomics_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.sex_emb = nn.Embedding(num_sex_classes, 4)
        self.loc_emb = nn.Embedding(num_loc_classes, 8)

        self.artifact_embeddings = nn.ModuleList([
            nn.Embedding(2, 2) for _ in range(num_artifact_classes)
        ])


        feature_dims = {
            'image': 256,
            'radiomics': 128,
            'clinical': 12+1,
            'artifacts': num_artifact_classes * 2
        }

        self.feature_dims = [feature_dims[m] for m in modality]
        self.total_dim = sum(self.feature_dims)

        if fusion_level == 'intermediate':
            if fusion_strategy in ['concat', 'weighted']:
                self.fusion_mlp = nn.Sequential(
                    nn.Linear(self.total_dim, 256),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(256, num_classes)
                )
            elif fusion_strategy == 'attention':
                self.projections = nn.ModuleDict({
                    mod: nn.Sequential(
                        nn.Linear(feature_dims[mod], max(self.feature_dims)),
                        nn.ReLU(),
                        nn.BatchNorm1d(max(self.feature_dims))
                    ) for mod in self.modality
                })
                self.attention = AttentionFusion(max(self.feature_dims), len(self.modality))
                self.fusion_mlp = nn.Sequential(
                    nn.Linear(max(self.feature_dims), 256),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(256, num_classes)
                )
            if fusion_strategy == 'weighted':
                self.weights = nn.Parameter(torch.ones(len(modality)))

        elif fusion_level == 'late':
            if fusion_strategy == 'concat':
                self.fusion_mlp = nn.Sequential(
                    nn.Linear(len(modality) * num_classes, 128),
                    nn.ReLU(),
                    nn.Linear(128, num_classes)
                )
            elif fusion_strategy == 'weighted':
                self.weights = nn.Parameter(torch.ones(len(modality)))
            elif fusion_strategy == 'attention':
                self.attention = AttentionFusion_Late(len(modality), num_classes=num_classes)
        
            self.modality_heads = nn.ModuleDict()

            if 'image' in modality:
                self.modality_heads['image'] = nn.Linear(256, num_classes)
            if 'radiomics' in modality:
                self.modality_heads['radiomics'] = nn.Linear(128, num_classes)
            if 'clinical' in modality:
                self.modality_heads['clinical'] = nn.Linear(12+1, num_classes)
            if 'artifacts' in modality:
                self.modality_heads['artifacts'] = nn.Linear(num_artifact_classes * 2, num_classes)





    def forward(self, image=None, radiomics=None, age=None, sex=None, loc=None, artifacts=None):
        features = []
        logits = []

        if 'image' in self.modality:
            image_feat = self.image_model(image)
            image_feat = self.image_proj(image_feat)
            if self.fusion_level == 'intermediate':
                features.append(image_feat)
            else:
                logits.append(self.modality_heads['image'](image_feat))

        if 'radiomics' in self.modality:
            rad_feat = self.radiomics_mlp(radiomics)
            if self.fusion_level == 'intermediate':
                features.append(rad_feat)
            else:
                logits.append(self.modality_heads['radiomics'](rad_feat))

        if 'clinical' in self.modality:
            age_feat = age.unsqueeze(1)  # Assuming age is a single scalar
            sex_feat = self.sex_emb(sex)
            loc_feat = self.loc_emb(loc)
            clinical_feat = torch.cat([age_feat, sex_feat, loc_feat], dim=1)
            if self.fusion_level == 'intermediate':
                features.append(clinical_feat)
            else:
                logits.append(self.modality_heads['clinical'](clinical_feat))

        if 'artifacts' in self.modality:
            art_feats = [self.artifact_embeddings[i](artifacts[:, i]) for i in range(artifacts.size(1))]
            artifact_feat = torch.cat(art_feats, dim=1)
            if self.fusion_level == 'intermediate':
                features.append(artifact_feat)
            else:
                logits.append(self.modality_heads['artifacts'](artifact_feat))


        if self.fusion_level == 'intermediate':
            if self.fusion_strategy == 'concat':
                fused = torch.cat(features, dim=1)
            elif self.fusion_strategy == 'weighted':
                norm_weights = F.softmax(self.weights, dim=0)
                weighted_features = [w * f for w, f in zip(norm_weights, features)]
                fused = torch.cat(weighted_features, dim=1)  # (B, 256+128+32+32)
            elif self.fusion_strategy == 'attention':
                features = [self.projections[m](f) for m, f in zip(self.modality, features)]
                fused = self.attention(features)
            return self.fusion_mlp(fused)

        elif self.fusion_level == 'late':
            if self.fusion_strategy == 'concat':
                stacked_preds = torch.stack(logits, dim=1)
                fused_logits = torch.sum(stacked_preds, dim=1)
            elif self.fusion_strategy == 'weighted':
                norm_weights = F.softmax(self.weights, dim=0)
                fused_logits = torch.stack([w * z for w, z in zip(norm_weights, logits)], dim=0).sum(dim=0)
            elif self.fusion_strategy == 'attention':
                # TODO fix
                fused_logits = self.attention(logits)
            return fused_logits
