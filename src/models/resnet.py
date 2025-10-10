import torch
from torchvision.models import resnet18, ResNet18_Weights
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryAUROC
)
import pytorch_lightning as L
from torchvision import transforms

image_size = 224

RESNET_INPUT_MEAN = [0.485, 0.456, 0.406]
RESNET_INPUT_SD = [0.229, 0.224, 0.225]

# Default image transformations
DEFAULT_DATA_TRANSFORMS = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(RESNET_INPUT_MEAN, RESNET_INPUT_SD) #mean and std dev values for each channel from ImageNet (pretrain data)
    ]),
    'val': transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(RESNET_INPUT_MEAN, RESNET_INPUT_SD) #mean and std dev values for each channel from ImageNet (pretrain data)
    ]),
    'test': transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(RESNET_INPUT_MEAN, RESNET_INPUT_SD) #mean and std dev values for each channel from ImageNet (pretrain data)
    ]),
}

class ResNetClassifier(L.LightningModule):
    def __init__(self, 
                in_channels=3,
                image_size=224,
                freeze_features=False, 
                lr=1e-3,
                weights=ResNet18_Weights.DEFAULT):
        super().__init__()
        self.save_hyperparameters()

        num_classes = 1  # binary classification
        self.model = resnet18(weights=weights)
        if in_channels <= 0: raise ValueError("in_channels must be a positive integer")
        #  Adapt first conv layer for variable channel count ---
        if in_channels != 3:
            old_conv = self.model.conv1
            new_conv = torch.nn.Conv2d(in_channels,
                                 old_conv.out_channels,
                                 kernel_size=old_conv.kernel_size,
                                 stride=old_conv.stride,
                                 padding=old_conv.padding,
                                 bias=(old_conv.bias is not None))
            with torch.no_grad():
                if in_channels == 1:
                    # average pretrained RGB weights to single channel
                    new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
                elif in_channels > 3:
                    # copy RGB weights to first 3 channels, init extras
                    new_conv.weight[:, :3, :, :] = old_conv.weight
                    extra = old_conv.weight.mean(dim=1, keepdim=True)
                    new_conv.weight[:, 3:, :, :] = extra.repeat(1, in_channels - 3, 1, 1)
                elif in_channels == 2:
                    new_conv.weight[:, :in_channels, :, :] = old_conv.weight[:, :in_channels, :, :]
                    # initialize any remaining channels with mean
                    mean_extra = old_conv.weight.mean(dim=1, keepdim=True)
                    new_conv.weight[:, in_channels:, :, :] = mean_extra.repeat(1, 3 - in_channels, 1, 1)
            self.model.conv1 = new_conv

        # Replace final layer
        in_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(in_features, num_classes)

        # Optionally freeze backbone
        if freeze_features:
            for name, param in self.model.named_parameters():
                if not name.startswith("fc."):
                    param.requires_grad = False


        # Loss
        self.criterion = torch.nn.BCEWithLogitsLoss()

        # Metrics
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()

        self.train_prec = BinaryPrecision()
        self.val_prec   = BinaryPrecision()
        self.test_prec  = BinaryPrecision()

        self.train_rec = BinaryRecall()
        self.val_rec   = BinaryRecall()
        self.test_rec  = BinaryRecall()

        self.train_f1 = BinaryF1Score()
        self.val_f1   = BinaryF1Score()
        self.test_f1  = BinaryF1Score()

        self.train_auc = BinaryAUROC()
        self.val_auc   = BinaryAUROC()
        self.test_auc  = BinaryAUROC()


    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage):
        x, y = batch
        logits = self(x).squeeze()
        loss = self.criterion(logits, y.float())
        preds = torch.sigmoid(logits)

        # metrics
        for metric_name in ["acc", "prec", "rec", "f1", "auc"]:
            metric = getattr(self, f"{stage}_{metric_name}")
            metric.update(preds, y.int())
            self.log(f"{stage}_{metric_name}", metric, on_epoch=True, prog_bar=True)

        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)