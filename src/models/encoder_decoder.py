import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryAUROC
)
import pytorch_lightning as L
from torchvision import transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

# from adapters.datasets.sida import SidADataModule

image_size = 224

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model

    def forward(self, x):
        return self.l1(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def forward(self, x):
        return self.l1(x)


class DisentanglementEncoder(nn.Module):
    """Encoder that disentangles features into content, specific, and common components"""
    def __init__(self, input_channels=3, hidden_dim=512):
        super().__init__()
        
        # Shared feature extractor
        self.shared_conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),  # 112x112
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),  # 56x56
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, 2, 1),  # 28x28
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(7)  # 7x7
        )
        
        # Three separate heads for disentanglement
        self.content_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.specific_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 4)
        )
        
        self.common_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 4)
        )
    
    def forward(self, x):
        # Extract shared features
        shared_feat = self.shared_conv(x)
        shared_feat = shared_feat.view(shared_feat.size(0), -1)
        
        # Disentangle into three components
        content = self.content_head(shared_feat)
        specific = self.specific_head(shared_feat)
        common = self.common_head(shared_feat)
        
        return content, specific, common

class RecombinationModule(nn.Module):
    """Recombines disentangled features for classification"""
    def __init__(self, content_dim=256, specific_dim=128, common_dim=128, num_classes=2):
        super().__init__()
        
        total_dim = content_dim + specific_dim + common_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Attention mechanism for feature weighting
        self.attention = nn.Sequential(
            nn.Linear(total_dim, max(total_dim // 4, 64)),  # Ensure minimum size
            nn.ReLU(),
            nn.Linear(max(total_dim // 4, 64), total_dim),
            nn.Sigmoid()
        )
    
    def forward(self, content, specific, common):
        # Debug prints to identify dimension issues
        # print(f"Content shape: {content.shape}")
        # print(f"Specific shape: {specific.shape}")  
        # print(f"Common shape: {common.shape}")
        
        # Combine features
        combined = torch.cat([content, specific, common], dim=1)
        # print(f"Combined shape: {combined.shape}")
        
        # Apply attention
        att_weights = self.attention(combined)
        weighted_features = combined * att_weights
        
        # Classification
        logits = self.classifier(weighted_features)
        return logits

class DisentanglementLoss(nn.Module):
    """Custom loss for disentanglement learning"""
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.1):
        super().__init__()
        self.alpha = alpha  # Classification loss weight
        self.beta = beta    # Orthogonality loss weight
        self.gamma = gamma  # Mutual information loss weight
        
        self.ce_loss = nn.CrossEntropyLoss()
    
    def orthogonality_loss(self, feat1, feat2):
        """Encourage orthogonality between different feature types"""
        # Handle different feature dimensions by projecting to same space
        min_dim = min(feat1.size(1), feat2.size(1))
        
        feat1_proj = feat1[:, :min_dim]
        feat2_proj = feat2[:, :min_dim]
        
        feat1_norm = F.normalize(feat1_proj, p=2, dim=1)
        feat2_norm = F.normalize(feat2_proj, p=2, dim=1)
        correlation = torch.abs(torch.sum(feat1_norm * feat2_norm, dim=1))
        return torch.mean(correlation)
    
    def mutual_info_loss(self, specific_feats, labels):
        """Encourage specific features to be discriminative"""
        # Ensure labels are long type
        labels = labels.long()
        
        # Simple implementation using variance
        fake_mask = (labels == 1)
        real_mask = (labels == 0)
        
        if fake_mask.sum() > 1 and real_mask.sum() > 1:
            fake_var = torch.var(specific_feats[fake_mask], dim=0).mean()
            real_var = torch.var(specific_feats[real_mask], dim=0).mean()
            return -torch.log(fake_var + real_var + 1e-8)
        return torch.tensor(0.0, device=specific_feats.device)
    
    def forward(self, logits, labels, content, specific, common):
        # Ensure labels are Long type for CrossEntropyLoss
        labels = labels.long()
        
        # Classification loss
        cls_loss = self.ce_loss(logits, labels)
        
        # Orthogonality losses (encourage different feature types to be independent)
        orth_loss1 = self.orthogonality_loss(content, specific)
        orth_loss2 = self.orthogonality_loss(content, common)
        orth_loss3 = self.orthogonality_loss(specific, common)
        orth_loss = (orth_loss1 + orth_loss2 + orth_loss3) / 3
        
        # Mutual information loss (encourage specific features to be discriminative)
        mi_loss = self.mutual_info_loss(specific, labels)
        
        total_loss = self.alpha * cls_loss + self.beta * orth_loss + self.gamma * mi_loss
        
        return {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'orth_loss': orth_loss,
            'mi_loss': mi_loss
        }

class DisentanglementDeepfakeDetector(L.LightningModule):
    """Main PyTorch Lightning module for disentanglement-based deepfake detection"""
    
    def __init__(self, input_channels=3, hidden_dim=512, learning_rate=1e-3, 
                 loss_weights={'alpha': 1.0, 'beta': 0.1, 'gamma': 0.1}):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize components
        self.encoder = DisentanglementEncoder(input_channels, hidden_dim)
        # Calculate actual dimensions from encoder output
        content_dim = hidden_dim // 2  # 256
        specific_dim = hidden_dim // 4  # 128  
        common_dim = hidden_dim // 4   # 128
        
        self.recombination = RecombinationModule(
            content_dim=content_dim, 
            specific_dim=specific_dim, 
            common_dim=common_dim
        )
        self.criterion = DisentanglementLoss(**loss_weights)
        
        # Metrics storage
        self.train_acc = []
        self.val_acc = []
    
    def forward(self, x):
        content, specific, common = self.encoder(x)
        logits = self.recombination(content, specific, common)
        return logits, content, specific, common
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        
        # Ensure proper data types
        images = images.float()
        labels = labels.long()
        
        logits, content, specific, common = self(images)
        
        # Calculate losses
        loss_dict = self.criterion(logits, labels, content, specific, common)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        # Log metrics
        self.log('train_loss', loss_dict['total_loss'], prog_bar=True)
        self.log('train_cls_loss', loss_dict['cls_loss'])
        self.log('train_orth_loss', loss_dict['orth_loss'])
        self.log('train_mi_loss', loss_dict['mi_loss'])
        self.log('train_acc', acc, prog_bar=True)
        
        return loss_dict['total_loss']
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        
        # Ensure proper data types
        images = images.float()
        labels = labels.long()
        
        logits, content, specific, common = self(images)
        
        # Calculate losses
        loss_dict = self.criterion(logits, labels, content, specific, common)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        # Log metrics
        self.log('val_loss', loss_dict['total_loss'], prog_bar=True)
        self.log('val_cls_loss', loss_dict['cls_loss'])
        self.log('val_acc', acc, prog_bar=True)
        
        return {'val_loss': loss_dict['total_loss'], 'val_acc': acc}
    
    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits, _, _, _ = self(images)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log('test_acc', acc)
        return {'test_acc': acc}
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.8)
        return [optimizer], [scheduler]
    
    def get_feature_representations(self, x):
        """Extract disentangled representations for analysis"""
        with torch.no_grad():
            content, specific, common = self.encoder(x)
        return {
            'content': content.cpu().numpy(),
            'specific': specific.cpu().numpy(), 
            'common': common.cpu().numpy()
        }

# Training example
# if __name__ == "__main__":
#     # Initialize model
#     model = DisentanglementDeepfakeDetector(
#         input_channels=3,
#         hidden_dim=512,
#         learning_rate=1e-3,
#         loss_weights={'alpha': 1.0, 'beta': 0.1, 'gamma': 0.1}
#     )
    
#     # Initialize data module (you'll need to implement this)
#     sida_data_module = SidADataModule(seed=42, transforms=DEFAULT_DATA_TRANSFORMS, additional_transforms=ela)

#     # Initialize trainer
#     trainer = L.Trainer(
#         max_epochs=50,
#         accelerator='gpu' if torch.cuda.is_available() else 'cpu',
#         devices=1,
#         log_every_n_steps=10,
#         check_val_every_n_epoch=1
#     )
    
#     # Train the model
#     trainer.fit(model, sida_data_module)
    
#     # Test the model
#     trainer.test(model, sida_data_module)
    
#     print("Disentanglement learning model setup complete!")
#     print("Key components:")
#     print("- Encoder: Disentangles features into content, specific, and common")
#     print("- Recombination: Combines features with attention for classification")  
#     print("- Custom loss: Balances classification, orthogonality, and mutual information")