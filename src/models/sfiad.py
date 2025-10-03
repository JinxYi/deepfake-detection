import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import (
    Accuracy,
    F1Score,
    AUROC
)
import timm

class SFFIModule(nn.Module):
    """Spatial-Frequency Feature Integration Module"""
    
    def __init__(self, epsilon=1e-8):
        super(SFFIModule, self).__init__()
        self.epsilon = epsilon
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W) where C=3 (RGB image)
        Returns:
            Combined spatial-frequency features of shape (B, C+1, H, W)
        """
        # Convert to grayscale by averaging across channels
        gray = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Apply 2D FFT
        fft_result = torch.fft.fft2(gray.squeeze(1))  # (B, H, W)
        

        # FFT shift to center the spectrum
        fft_shifted = torch.fft.fftshift(fft_result)
        
        # Compute amplitude spectrum
        amplitude = torch.abs(fft_shifted)
        
        # Apply logarithmic transformation
        amplitude_log = torch.log(amplitude + self.epsilon)
        
        # Normalize to [0, 1]
        amplitude_norm = amplitude_log / (torch.max(amplitude_log.view(amplitude_log.size(0), -1), dim=1)[0].view(-1, 1, 1) + self.epsilon)
        
        # Add channel dimension back
        amplitude_norm = amplitude_norm.unsqueeze(1)  # (B, 1, H, W)
        
        # Concatenate with original color image
        combined = torch.cat([x, amplitude_norm], dim=1)  # (B, 4, H, W)
        
        return combined


class ModifiedXception(nn.Module):
    """Modified Xception backbone with custom input channels"""
    
    def __init__(self, input_channels=4, pretrained=True):
        super(ModifiedXception, self).__init__()        
        self.model = timm.create_model('xception', pretrained=pretrained, num_classes=0)
    
        # Modify the first conv layer to accept 4 channels instead of 3
        original_conv = self.model.conv1
        
        # Create new conv layer with desired input channels
        self.model.conv1 = nn.Conv2d(
            input_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        # If pretrained, initialize new layer intelligently
        if pretrained:
            with torch.no_grad():
                # Copy weights from original 3 channels
                self.model.conv1.weight[:, :3, :, :] = original_conv.weight
                
                # Initialize the 4th channel (frequency channel) as average of RGB channels
                self.model.conv1.weight[:, 3:, :, :] = original_conv.weight.mean(dim=1, keepdim=True)
                
                # Copy bias if it exists
                if original_conv.bias is not None:
                    self.model.conv1.bias = original_conv.bias
    
    def forward(self, x):
        return self.model(x)


class FrameLevelClassifier(nn.Module):
    """Frame-level classifier"""
    def __init__(self, input_dim=2048, num_classes=2):
        super(FrameLevelClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)


class AAMLLoss(nn.Module):
    """Authenticity-Aware Margin Loss"""
    
    def __init__(self, num_classes=2, scale=30.0, m_max=0.5):
        super(AAMLLoss, self).__init__()
        self.num_classes = num_classes
        self.scale = scale
        self.m_max = m_max
        self.class_counts = None
    
    def update_class_counts(self, labels):
        """Update class sample counts from the batch"""
        unique, counts = torch.unique(labels, return_counts=True)
        if self.class_counts is None:
            self.class_counts = torch.zeros(self.num_classes, device=labels.device)
        for cls, count in zip(unique, counts):
            self.class_counts[cls] += count
    
    def compute_margins(self):
        """Compute dynamic margins based on class counts"""
        if self.class_counts is None:
            return torch.zeros(self.num_classes)
        
        # m_y = (1 / sqrt(n_y)) * (m_max / max(1 / sqrt(n_y)))
        inv_sqrt_counts = 1.0 / torch.sqrt(self.class_counts + 1e-8)
        margins = inv_sqrt_counts * (self.m_max / torch.max(inv_sqrt_counts))
        return margins
    
    def forward(self, logits, labels):
        """
        Args:
            logits: Model output logits (B, num_classes)
            labels: Ground truth labels (B,)
        """
        batch_size = logits.size(0)
        
        # Update class counts
        self.update_class_counts(labels)
        
        # Compute margins
        margins = self.compute_margins().to(logits.device)
        
        # Apply margin adjustment
        adjusted_logits = logits.clone()
        for i in range(batch_size):
            adjusted_logits[i, labels[i]] -= margins[labels[i]]
        
        # Scale logits
        adjusted_logits = self.scale * adjusted_logits
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(adjusted_logits, labels)
        
        return loss


class SFIADModel(pl.LightningModule):
    """Complete SFIAD Model with PyTorch Lightning"""
    
    def __init__(
        self,
        num_classes=2,
        learning_rate=2e-4,
        weight_decay=5e-4,
        scale=30.0,
        m_max=0.5
    ):
        super(SFIADModel, self).__init__()
        self.save_hyperparameters()
        
        # Model components
        self.sffi = SFFIModule()
        self.backbone = ModifiedXception(input_channels=4, pretrained=True)
        self.classifier = FrameLevelClassifier(input_dim=2048, num_classes=num_classes)
        
        # Loss function
        self.criterion = AAMLLoss(num_classes=num_classes, scale=scale, m_max=m_max)
        
        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_auroc = AUROC(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes)

        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_auroc = AUROC(task="multiclass", num_classes=num_classes)
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes)
    
    def forward(self, x):
        """
        Args:
            x: Input video frames (B, num_frames, C, H, W) or (B, C, H, W) for single frame
        """
        # Handle both video (multiple frames) and single frame inputs
        if x.dim() == 5:  # (B, num_frames, C, H, W)
            batch_size, num_frames = x.shape[:2]
            x = x.view(-1, *x.shape[2:])  # (B*num_frames, C, H, W)
        else:
            batch_size, num_frames = x.shape[0], 1
        
        # Apply SFFI
        x = self.sffi(x)
        
        # Extract features with backbone
        features = self.backbone(x)
        
        # Classify each frame
        logits = self.classifier(features)
        
        # Reshape back if needed
        if num_frames > 1:
            logits = logits.view(batch_size, num_frames, -1)
            # Average predictions across frames
            logits = logits.mean(dim=1)
        
        return logits
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)

        # Update metrics
        self.val_acc.update(preds, y)
        self.val_auroc.update(probs, y)
        self.val_f1.update(preds, y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)
        self.log('val_auroc', self.val_auroc, prog_bar=True)
        self.log('val_f1', self.val_f1, prog_bar=True)

        return {'val_loss': loss, 'val_acc': self.val_acc, 'val_auroc': self.val_auroc, 'val_f1': self.val_f1}
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)

        # Update metrics
        self.test_acc.update(preds, y)
        self.test_auroc.update(probs, y)
        self.test_f1.update(preds, y)

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', self.test_acc, prog_bar=True)
        self.log('test_auroc', self.test_auroc, prog_bar=True)
        self.log('test_f1', self.test_f1, prog_bar=True)

        return {'test_loss': loss, 'test_acc': self.test_acc, 'test_auroc': self.test_auroc, 'test_f1': self.test_f1}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999)
        )
        return optimizer


# Example usage
if __name__ == "__main__":
    # Initialize model
    model = SFIADModel(
        num_classes=2,
        learning_rate=2e-4,
        weight_decay=5e-4,
        scale=30.0,
        m_max=0.5
    )
    
    # Test forward pass
    batch_size = 4
    num_frames = 32
    channels = 3
    height, width = 256, 256
    
    # Single frame input
    x_single = torch.randn(batch_size, channels, height, width)
    output_single = model(x_single)
    print(f"Single frame output shape: {output_single.shape}")
    print(f"Single frame output: {output_single}")

    # Multi-frame (video) input
    x_video = torch.randn(batch_size, num_frames, channels, height, width)
    output_video = model(x_video)
    print(f"Video output shape: {output_video.shape}")
    print(f"Video output: {output_video}")