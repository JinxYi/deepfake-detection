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
    def __init__(self, freeze_features=False, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        num_classes = 1  # binary classification
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Optionally freeze backbone
        if freeze_features:
            for name, param in self.model.named_parameters():
                if not name.startswith("fc."):
                    param.requires_grad = False

        # Replace final layer
        in_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(in_features, num_classes)

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

        # Update metrics
        acc  = getattr(self, f"{stage}_acc")
        prec = getattr(self, f"{stage}_prec")
        rec  = getattr(self, f"{stage}_rec")
        f1   = getattr(self, f"{stage}_f1")
        auc  = getattr(self, f"{stage}_auc")

        acc.update(preds, y.int())
        prec.update(preds, y.int())
        rec.update(preds, y.int())
        f1.update(preds, y.int())
        auc.update(preds, y.int())

        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_prec", prec, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_rec", rec, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_f1", f1, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_auc", auc, prog_bar=True, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)