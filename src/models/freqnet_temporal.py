import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchmetrics

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class FreqNetBackbone(nn.Module):
    """Modified FreqNet that returns features instead of final classification"""
    
    def __init__(self, block=Bottleneck, layers=[3, 4]):
        super(FreqNetBackbone, self).__init__()
        
        # Initialize frequency domain weights - remove .cuda() for device flexibility
        self.weight1 = nn.Parameter(torch.randn((64, 3, 1, 1)))
        self.bias1   = nn.Parameter(torch.randn((64,)))
        self.realconv1 = conv1x1(64, 64, stride=1)
        self.imagconv1 = conv1x1(64, 64, stride=1)

        self.weight2 = nn.Parameter(torch.randn((64, 64, 1, 1)))
        self.bias2   = nn.Parameter(torch.randn((64,)))
        self.realconv2 = conv1x1(64, 64, stride=1)
        self.imagconv2 = conv1x1(64, 64, stride=1)

        self.weight3 = nn.Parameter(torch.randn((256, 256, 1, 1)))
        self.bias3   = nn.Parameter(torch.randn((256,)))
        self.realconv3 = conv1x1(256, 256, stride=1)
        self.imagconv3 = conv1x1(256, 256, stride=1)

        self.weight4 = nn.Parameter(torch.randn((256, 256, 1, 1)))
        self.bias4   = nn.Parameter(torch.randn((256,)))
        self.realconv4 = conv1x1(256, 256, stride=1)
        self.imagconv4 = conv1x1(256, 256, stride=1)
        
        self.inplanes = 64 
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def hfreqWH(self, x, scale):
        """High-frequency representation in Width-Height dimensions"""
        assert scale > 2
        x = torch.fft.fft2(x, norm="ortho")
        x = torch.fft.fftshift(x, dim=[-2, -1]) 
        b, c, h, w = x.shape
        x[:, :, h//2-h//scale:h//2+h//scale, w//2-w//scale:w//2+w//scale] = 0.0
        x = torch.fft.ifftshift(x, dim=[-2, -1])
        x = torch.fft.ifft2(x, norm="ortho")
        x = torch.real(x)
        x = F.relu(x, inplace=True)
        return x

    def hfreqC(self, x, scale):
        """High-frequency representation in Channel dimension"""
        assert scale > 2
        x = torch.fft.fft(x, dim=1, norm="ortho")
        x = torch.fft.fftshift(x, dim=1) 
        b, c, h, w = x.shape
        x[:, c//2-c//scale:c//2+c//scale, :, :] = 0.0
        x = torch.fft.ifftshift(x, dim=1)
        x = torch.fft.ifft(x, dim=1, norm="ortho")
        x = torch.real(x)
        x = F.relu(x, inplace=True)
        return x
        
    def forward(self, x):
        # HFRI - High Frequency Representation of Image
        x = self.hfreqWH(x, 4)
        x = F.conv2d(x, self.weight1, self.bias1, stride=1, padding=0)
        x = F.relu(x, inplace=True)
        
        # HFRFC - High Frequency Representation Feature Channel
        x = self.hfreqC(x, 4)
        
        # FCL - Frequency Conv Layer
        x = torch.fft.fft2(x, norm="ortho")
        x = torch.fft.fftshift(x, dim=[-2, -1]) 
        x = torch.complex(self.realconv1(x.real), self.imagconv1(x.imag))
        x = torch.fft.ifftshift(x, dim=[-2, -1])
        x = torch.fft.ifft2(x, norm="ortho")
        x = torch.real(x)
        x = F.relu(x, inplace=True)

        # HFRFS - High Frequency Representation Feature Spatial
        x = self.hfreqWH(x, 4)
        x = F.conv2d(x, self.weight2, self.bias2, stride=2, padding=0)
        x = F.relu(x, inplace=True)

        # HFRFC
        x = self.hfreqC(x, 4)
        
        # FCL
        x = torch.fft.fft2(x, norm="ortho")
        x = torch.fft.fftshift(x, dim=[-2, -1]) 
        x = torch.complex(self.realconv2(x.real), self.imagconv2(x.imag))
        x = torch.fft.ifftshift(x, dim=[-2, -1])
        x = torch.fft.ifft2(x, norm="ortho")
        x = torch.real(x)
        x = F.relu(x, inplace=True)

        x = self.maxpool(x)
        x = self.layer1(x)  # 64 -> 256 channels
        
        # HFRFS
        x = self.hfreqWH(x, 4)
        x = F.conv2d(x, self.weight3, self.bias3, stride=1, padding=0)
        x = F.relu(x, inplace=True)
        
        # FCL
        x = torch.fft.fft2(x, norm="ortho")
        x = torch.fft.fftshift(x, dim=[-2, -1]) 
        x = torch.complex(self.realconv3(x.real), self.imagconv3(x.imag))
        x = torch.fft.ifftshift(x, dim=[-2, -1])
        x = torch.fft.ifft2(x, norm="ortho")
        x = torch.real(x)
        x = F.relu(x, inplace=True)

        # HFRFS
        x = self.hfreqWH(x, 4)
        x = F.conv2d(x, self.weight4, self.bias4, stride=2, padding=0)
        x = F.relu(x, inplace=True)

        # FCL
        x = torch.fft.fft2(x, norm="ortho")
        x = torch.fft.fftshift(x, dim=[-2, -1]) 
        x = torch.complex(self.realconv4(x.real), self.imagconv4(x.imag))
        x = torch.fft.ifftshift(x, dim=[-2, -1])
        x = torch.fft.ifft2(x, norm="ortho")
        x = torch.real(x)
        x = F.relu(x, inplace=True)

        x = self.layer2(x)  # 256 -> 512 channels
        x = self.avgpool(x)  # Global average pooling
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, 512)
        
        return x


class TemporalFreqNet(pl.LightningModule):
    """Temporal extension of FreqNet using PyTorch Lightning"""
    
    def __init__(self, 
                 num_frames=16, 
                 temporal_method='lstm',
                 hidden_size=256,
                 num_layers=2,
                 dropout=0.1,
                 learning_rate=1e-3,
                 num_classes=1):
        super().__init__()
        
        self.save_hyperparameters()
        self.num_frames = num_frames
        self.temporal_method = temporal_method
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        
        # FreqNet backbone for feature extraction
        self.freqnet_backbone = FreqNetBackbone()
        
        # Temporal aggregation modules
        if temporal_method == 'lstm':
            self.temporal_module = nn.LSTM(
                input_size=512,  # FreqNet output size
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=True
            )
            classifier_input_size = hidden_size * 2  # Bidirectional
            
        elif temporal_method == 'gru':
            self.temporal_module = nn.GRU(
                input_size=512,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=True
            )
            classifier_input_size = hidden_size * 2
            
        elif temporal_method == 'attention':
            self.temporal_module = nn.MultiheadAttention(
                embed_dim=512,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            classifier_input_size = 512
            
        elif temporal_method == 'conv3d':
            # 3D convolution approach
            self.temporal_module = nn.Sequential(
                nn.Conv3d(512, 256, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
                nn.ReLU(inplace=True),
                nn.Conv3d(256, 128, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool3d((1, 1, 1))
            )
            classifier_input_size = 128
            
        else:  # 'average' pooling
            self.temporal_module = nn.Identity()
            classifier_input_size = 512
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_input_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        # Metrics
        self.train_acc = torchmetrics.Accuracy(task='binary' if num_classes == 1 else 'multiclass', 
                                             num_classes=None if num_classes == 1 else num_classes)
        self.val_acc = torchmetrics.Accuracy(task='binary' if num_classes == 1 else 'multiclass',
                                           num_classes=None if num_classes == 1 else num_classes)
        self.test_acc = torchmetrics.Accuracy(task='binary' if num_classes == 1 else 'multiclass',
                                            num_classes=None if num_classes == 1 else num_classes)
        
    def forward(self, x):
        # Input: (batch_size, num_frames, channels, height, width)
        batch_size, num_frames, c, h, w = x.shape
        
        # Extract features from each frame
        frame_features = []
        for i in range(num_frames):
            frame = x[:, i]  # (batch_size, c, h, w)
            features = self.freqnet_backbone(frame)  # (batch_size, 512)
            frame_features.append(features)
        
        # Stack features: (batch_size, num_frames, 512)
        temporal_features = torch.stack(frame_features, dim=1)
        
        # Apply temporal aggregation
        if self.temporal_method in ['lstm', 'gru']:
            temporal_output, _ = self.temporal_module(temporal_features)
            # Use last output for classification
            aggregated_features = temporal_output[:, -1]  # (batch_size, hidden_size*2)
            
        elif self.temporal_method == 'attention':
            # Self-attention over temporal dimension
            attended_features, _ = self.temporal_module(
                temporal_features, temporal_features, temporal_features
            )
            # Average over time dimension
            aggregated_features = attended_features.mean(dim=1)  # (batch_size, 512)
            
        elif self.temporal_method == 'conv3d':
            # Reshape for 3D conv: (batch_size, channels, time, height, width)
            # Since we already have flattened features, we need to reshape
            temporal_features = temporal_features.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
            # (batch_size, 512, num_frames, 1, 1)
            conv_output = self.temporal_module(temporal_features)
            aggregated_features = conv_output.view(batch_size, -1)
            
        else:  # Average pooling
            aggregated_features = temporal_features.mean(dim=1)  # (batch_size, 512)
        
        # Final classification
        logits = self.classifier(aggregated_features)
        return logits
    
    def training_step(self, batch, batch_idx):
        videos, labels = batch
        logits = self(videos)
        
        if self.num_classes == 1:
            loss = F.binary_cross_entropy_with_logits(logits.squeeze(), labels.float())
            preds = torch.sigmoid(logits.squeeze()) > 0.5
        else:
            loss = F.cross_entropy(logits, labels)
            preds = torch.argmax(logits, dim=1)
        
        self.train_acc(preds, labels)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        videos, labels = batch
        logits = self(videos)
        
        if self.num_classes == 1:
            loss = F.binary_cross_entropy_with_logits(logits.squeeze(), labels.float())
            preds = torch.sigmoid(logits.squeeze()) > 0.5
        else:
            loss = F.cross_entropy(logits, labels)
            preds = torch.argmax(logits, dim=1)
        
        self.val_acc(preds, labels)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        videos, labels = batch
        logits = self(videos)
        
        if self.num_classes == 1:
            preds = torch.sigmoid(logits.squeeze()) > 0.5
        else:
            preds = torch.argmax(logits, dim=1)
        
        self.test_acc(preds, labels)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=100)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }


# Example usage and training script
def create_model(temporal_method='lstm', num_frames=16):
    """Factory function to create different temporal models"""
    
    model = TemporalFreqNet(
        num_frames=num_frames,
        temporal_method=temporal_method,
        hidden_size=256,
        num_layers=2,
        dropout=0.1,
        learning_rate=1e-3,
        num_classes=1  # Binary classification: real vs fake
    )
    
    return model


# Training example (you would replace this with your actual data loading)
if __name__ == "__main__":
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    from pytorch_lightning.loggers import TensorBoardLogger
    
    # Create model
    model = create_model(temporal_method='lstm', num_frames=16)
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        save_top_k=1,
        filename='temporal-freqnet-{epoch:02d}-{val_acc:.2f}'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min'
    )
    
    # Logger
    logger = TensorBoardLogger('lightning_logs', name='temporal_freqnet')
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision=16  # Mixed precision for efficiency
    )
    
    # You would add your actual data loaders here
    # trainer.fit(model, train_dataloader, val_dataloader)
    print("Model created successfully! Add your data loaders to start training.")