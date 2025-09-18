"""
PyTorch Lightning implementation skeleton for the UCF (Uncovering Common Features)
model from ICCV2023 adapted from the paper and an available PyTorch implementation.

This file contains:
 - UCFLightningModule : the LightningModule implementing encoders, decoder (AdaIN), heads,
   losses and training_step/validation_step.
 - UCFDataModule : a DataModule skeleton you can fill with your dataset loaders.

Notes:
 - This is a focused, readable skeleton. You will need to plug in datasets and may
   want to replace the backbone with a faithful Xception implementation (e.g. via
   timm or a local xception.py). The code is structured so swapping the backbone is
   straightforward.
 - The implementation aims to follow the paper structure (content encoder, fingerprint
   encoder, AdaIN decoder, common/specific heads, reconstruction & contrastive losses).

Dependencies:
  - torch, torchvision
  - pytorch_lightning
  - optionally timm (for xception/other backbones)

Run example (toy):
    python ucf_pytorch_lightning.py

"""

from typing import Optional, Tuple
import math
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# try to import timm for backbones; if not available, fallback to ResNet50
try:
    import timm
    _HAS_TIMM = True
except Exception:
    _HAS_TIMM = False


# ------------------------------- Utilities ---------------------------------
class AdaIN(nn.Module):
    """Adaptive Instance Normalization layer.
    Given content features x and style parameters (gamma, beta), apply
    normalized_x * gamma + beta.
    """
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor):
        # x: (B, C, H, W); gamma, beta: (B, C)
        B, C, H, W = x.shape
        # Instance norm
        mean = x.view(B, C, -1).mean(-1).view(B, C, 1, 1)
        var = x.view(B, C, -1).var(-1, unbiased=False).view(B, C, 1, 1)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        gamma = gamma.view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)
        return x_norm * gamma + beta


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=2, activate_final=False):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2 or activate_final:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ------------------------------- Decoder -----------------------------------
class DecoderWithAdaIN(nn.Module):
    """A simple decoder that upsamples and applies AdaIN conditioning using
    fingerprint-style vectors. This is a compact, understandable template — not an
    exact reproduction of the paper's decoder. The decoder receives content features
    (in spatial form) and style vectors (common and specific) and uses AdaIN to
    inject them.
    """
    def __init__(self, in_channels=512, style_dim=256, out_channels=3):
        super().__init__()
        # we'll progressively upsample to 256x256 assuming input spatial size is small
        # Build a small conv block sequence with AdaIN at each block
        self.adain = AdaIN()
        # Produce gamma/beta from style vector
        self.style_mlp = nn.Sequential(
            nn.Linear(style_dim, style_dim),
            nn.ReLU(inplace=True),
            nn.Linear(style_dim, in_channels * 2),  # gamma & beta per channel
        )

        # simple upsampling conv blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels // 4, in_channels // 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(in_channels // 8, out_channels, kernel_size=3, padding=1)

    def forward(self, content_feat: torch.Tensor, style_vec: torch.Tensor):
        # style_vec: (B, style_dim)
        B, C, H, W = content_feat.shape
        style_params = self.style_mlp(style_vec)  # (B, 2*C)
        gamma, beta = style_params.chunk(2, dim=1)
        # AdaIN then upsample
        x = self.adain(content_feat, gamma, beta)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv1(x)
        # second AdaIN using same gamma/beta but resized to match smaller channels
        # To keep sample simple, reproject gamma/beta to match channel dims
        # (in practice you might want different style MLPs per block)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv3(x)
        x = self.out_conv(x)
        x = torch.sigmoid(x)  # image range [0,1]
        return x


# ------------------------------- Lightning Module --------------------------
class UCFLightningModule(pl.LightningModule):
    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        content_feat_dim: int = 512,
        fingerprint_dim: int = 256,
        num_specific_classes: int = 6,
        lambda_s: float = 0.1,
        lambda_rec: float = 0.3,
        lambda_con: float = 0.05,
        contrast_margin: float = 3.0,
        lr: float = 2e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ------------------ Backbones / encoders ------------------
        # We'll build a backbone feature extractor and then split head for
        # content encoder and fingerprint encoder (same architecture, separate params)
        self.backbone_name = backbone_name
        self.content_encoder = self._make_backbone(pretrained)
        self.fingerprint_encoder = self._make_backbone(pretrained)

        # Projectors: map backbone outputs to desired dims
        # We assume backbone returns a feature map; we add a conv to produce content features
        self.content_proj = nn.Conv2d(self._backbone_out_channels(), content_feat_dim, kernel_size=1)
        self.fingerprint_pool = nn.AdaptiveAvgPool2d(1)  # produce vector
        self.fingerprint_proj = nn.Linear(self._backbone_out_channels(), fingerprint_dim)

        # Further split fingerprint into common and specific by two small MLPs
        self.common_proj = nn.Sequential(nn.Linear(fingerprint_dim, fingerprint_dim // 2), nn.ReLU(), nn.Linear(fingerprint_dim // 2, fingerprint_dim // 2))
        self.specific_proj = nn.Sequential(nn.Linear(fingerprint_dim, fingerprint_dim // 2), nn.ReLU(), nn.Linear(fingerprint_dim // 2, fingerprint_dim // 2))

        # ------------------ Heads ------------------
        # Common forgery head (binary)
        self.common_head = nn.Sequential(
            nn.Linear(fingerprint_dim // 2, fingerprint_dim // 4),
            nn.ReLU(),
            nn.Linear(fingerprint_dim // 4, 2),
        )
        # Specific forgery head (multi-class: real + K specific methods)
        self.specific_head = nn.Sequential(
            nn.Linear(fingerprint_dim // 2, fingerprint_dim // 4),
            nn.ReLU(),
            nn.Linear(fingerprint_dim // 4, num_specific_classes),
        )

        # ------------------ Decoder ------------------
        # decoder expects content features of shape (B, content_feat_dim, h, w)
        self.decoder = DecoderWithAdaIN(in_channels=content_feat_dim, style_dim=(fingerprint_dim // 2), out_channels=3)

        # ------------------ Loss hyperparams ------------------
        self.lambda_s = lambda_s
        self.lambda_rec = lambda_rec
        self.lambda_con = lambda_con
        self.contrast_margin = contrast_margin
        self.lr = lr

        # classification losses
        self.ce = nn.CrossEntropyLoss()
        # L1 for reconstruction
        self.l1 = nn.L1Loss()

    def _make_backbone(self, pretrained: bool):
        if _HAS_TIMM:
            # try to use xception if available
            if 'xception' in self.backbone_name.lower():
                model = timm.create_model('xception', pretrained=pretrained, features_only=True)
            else:
                model = timm.create_model(self.backbone_name, pretrained=pretrained, features_only=True)
            return model
        else:
            # fallback to torchvision resnet50 truncated
            from torchvision.models import resnet50
            backbone = resnet50(pretrained=pretrained)
            # remove fc and avgpool
            modules = list(backbone.children())[:-2]
            return nn.Sequential(*modules)

    def _backbone_out_channels(self):
        # if timm model with features_only True, the last entry channels may be found in feature_info
        if _HAS_TIMM:
            try:
                m = timm.create_model(self.backbone_name, pretrained=False, features_only=True)
                return m.feature_info.channels_list[-1]
            except Exception:
                return 2048
        else:
            return 2048

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return content feature map, common vector, specific vector
        content_feat: (B, Cc, H, W)
        common_vec, specific_vec: (B, Cf)
        """
        # content encoder path
        if _HAS_TIMM:
            feats = self.content_encoder(x)[-1]  # take last feature map
        else:
            feats = self.content_encoder(x)
        content_feat = self.content_proj(feats)  # (B, Cc, h, w)

        # fingerprint encoder path (vector)
        if _HAS_TIMM:
            fp_feats = self.fingerprint_encoder(x)[-1]
        else:
            fp_feats = self.fingerprint_encoder(x)
        pooled = self.fingerprint_pool(fp_feats).flatten(1)
        fp_vec = self.fingerprint_proj(pooled)

        common_vec = self.common_proj(fp_vec)
        specific_vec = self.specific_proj(fp_vec)
        return content_feat, common_vec, specific_vec

    # ------------------- Contrastive losses ---------------------------------
    def contrastive_loss(self, z: torch.Tensor, labels: torch.Tensor, margin: float):
        """Simple margin-based contrastive loss across pairs in batch.
        z: (B, D) features; labels: (B,) integer labels where same label = positive.
        We'll compute pairwise distances and apply loss as in paper style.
        This implementation is O(B^2) and intended for clarity not speed.
        """
        B = z.shape[0]
        if B < 2:
            return z.new_tensor(0.0)
        # pairwise squared euclidean distances
        dist_sq = torch.cdist(z, z, p=2.0) ** 2
        loss = z.new_tensor(0.0)
        count = 0
        for i in range(B):
            for j in range(i + 1, B):
                if labels[i] == labels[j]:
                    # positive pair -> minimize distance
                    loss = loss + dist_sq[i, j]
                else:
                    # negative pair -> push beyond margin
                    loss = loss + F.relu(margin - torch.sqrt(dist_sq[i, j] + 1e-8)) ** 2
                count += 1
        if count == 0:
            return z.new_tensor(0.0)
        return loss / count

    # ------------------- Training / validation steps -----------------------
    def training_step(self, batch, batch_idx):
        # batch expected to be a dict with 'img', 'is_fake' (0/1), 'method' (0..K-1)
        imgs = batch['img']  # (B,3,H,W)
        is_fake = batch['is_fake'].long()
        method = batch['method'].long()  # specific class labels

        content_feat, common_vec, specific_vec = self.encode(imgs)

        # heads
        common_logits = self.common_head(common_vec)
        specific_logits = self.specific_head(specific_vec)

        l_common = self.ce(common_logits, is_fake)
        l_specific = self.ce(specific_logits, method)

        # reconstruction: self and cross reconstruction
        # self-recon: reconstruct image using its own content & style (common+specific)
        style_vec = common_vec + specific_vec
        recon_self = self.decoder(content_feat, style_vec)
        l_recon_self = self.l1(recon_self, imgs)

        # cross-recon: shuffle batch and decode with mixed styles
        idx = torch.randperm(imgs.size(0))
        content_feat_shuffled = content_feat[idx]
        style_shuffled = (common_vec + specific_vec)[idx]
        # cross: use content from i and style from j
        recon_cross = self.decoder(content_feat, style_shuffled)
        l_recon_cross = self.l1(recon_cross, imgs)

        l_recon = l_recon_self + l_recon_cross

        # contrastive: encourage common vectors cluster by fake/real, and specific cluster by method
        l_con_common = self.contrastive_loss(common_vec, is_fake, self.contrast_margin)
        l_con_specific = self.contrastive_loss(specific_vec, method, self.contrast_margin)
        l_con = l_con_common + l_con_specific

        loss = l_common + self.lambda_s * l_specific + self.lambda_rec * l_recon + self.lambda_con * l_con

        # logs
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/l_common', l_common, on_step=True, on_epoch=True)
        self.log('train/l_specific', l_specific, on_step=True, on_epoch=True)
        self.log('train/l_recon', l_recon, on_step=True, on_epoch=True)
        self.log('train/l_con', l_con, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        imgs = batch['img']
        is_fake = batch['is_fake'].long()
        content_feat, common_vec, specific_vec = self.encode(imgs)
        common_logits = self.common_head(common_vec)
        l_common = self.ce(common_logits, is_fake)
        preds = torch.softmax(common_logits, dim=1)[:, 1]
        # log metrics
        self.log('val/l_common', l_common, on_step=False, on_epoch=True, prog_bar=True)
        # user can add AUC/accuracy metrics with torchmetrics or manual code
        return {'preds': preds.detach(), 'labels': is_fake.detach()}

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.5, 0.999))
        return opt


# --------------------- DataModule (skeleton) ------------------------------
class DummyFaceDataset(Dataset):
    """A tiny dummy dataset for smoke testing the training loop.
    Replace with your real dataset that yields dicts: {'img', 'is_fake', 'method'}
    """
    def __init__(self, length=100, image_size=256):
        self.length = length
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = torch.rand(3, self.image_size, self.image_size)
        is_fake = torch.randint(0, 2, (1,)).item()
        method = torch.randint(0, 6, (1,)).item()
        return {'img': img, 'is_fake': is_fake, 'method': method}


class UCFDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=8, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        # Replace these with real dataset loading
        self.train_dataset = DummyFaceDataset(length=200)
        self.val_dataset = DummyFaceDataset(length=50)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


# --------------------------- Run example ----------------------------------
if __name__ == "__main__":
    # quick smoke test to ensure module constructs and a single training step runs
    dm = UCFDataModule(batch_size=4)
    dm.setup()
    model = UCFLightningModule(backbone_name='resnet50', pretrained=False)

    trainer = pl.Trainer(max_epochs=1, devices=1 if torch.cuda.is_available() else None, accelerator='gpu' if torch.cuda.is_available() else 'cpu', logger=False)
    trainer.fit(model, datamodule=dm)
