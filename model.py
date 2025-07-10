import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class MultiModalNet(nn.Module):
    def __init__(self, num_sex_classes, num_loc_classes, num_artifact_classes, num_classes, radiomics_dim, num_artifacts):
        super().__init__()
        
        self.image_model = EfficientNet.from_pretrained('efficientnet-b3')
        self.image_model._fc = nn.Identity()
        image_emb_dim = 1536

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
        
        self.sex_emb = nn.Embedding(num_sex_classes, emb_dim=4)
        self.loc_emb = nn.Embedding(num_loc_classes, emb_dim=8)
        
        self.artifact_embeddings = nn.ModuleList([
            nn.Embedding(2, 2) for _ in range(5)
        ])
        
        fusion_dim = image_emb_dim + 128 + 4 + 8 + num_artifacts * 2
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )
        
    def forward(self, image, radiomics, sex, loc, artifacts):
        image_feat = self.image_model(image)  # (batch, 1536)
        
        radiomics_feat = self.radiomics_mlp(radiomics)  # (batch, 128)
        
        sex_feat = self.sex_emb(sex)  # (batch, 4)
        loc_feat = self.loc_emb(loc)  # (batch, 8)
        clinical_feat = torch.cat([sex_feat, loc_feat], dim=1)  # (batch, 12)
        
        artifacts_feat = torch.cat([self.artifact_embeddings[i](artifacts[:, i]) for i in range(artifacts.size(1))], dim=1)  # (batch, num_artifacts * 4)
        
        fused = torch.cat([image_feat, radiomics_feat, clinical_feat, artifacts_feat], dim=1)
        
        output = self.fusion_mlp(fused)
        return output
