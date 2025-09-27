# import numpy as np
# import torch
# import torch.nn as nn
# from torchvision.models import resnet18, ResNet18_Weights
# from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
# from torchmetrics.classification import (
#     BinaryAccuracy,
#     BinaryPrecision,
#     BinaryRecall,
#     BinaryF1Score,
#     BinaryAUROC
# )
# from scipy import optimize
# import pytorch_lightning as L
# from torchvision import transforms
# from torch.optim import Adam
# from torch.optim.lr_scheduler import StepLR
# import torch.nn.functional as F
# from loss_functions.balance_loss import LDAMLoss
# from src.loss_functions.regularization_loss import ContrastiveLoss 

# image_size = 224

# class Conv2d1x1(nn.Module):
#     def __init__(self, in_f, hidden_dim, out_f):
#         super(Conv2d1x1, self).__init__()
#         self.conv2d = nn.Sequential(nn.Conv2d(in_f, hidden_dim, 1, 1),
#                                     nn.LeakyReLU(inplace=True),
#                                     nn.Conv2d(hidden_dim, hidden_dim, 1, 1),
#                                     nn.LeakyReLU(inplace=True),
#                                     nn.Conv2d(hidden_dim, out_f, 1, 1),)

#     def forward(self, x):
#         x = self.conv2d(x)
#         return x
    
# class Head(nn.Module):
#     def __init__(self, in_f, hidden_dim, out_f):
#         super(Head, self).__init__()
#         self.do = nn.Dropout(0.2)
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.mlp = nn.Sequential(nn.Linear(in_f, hidden_dim),
#                                  nn.LeakyReLU(inplace=True),
#                                  nn.Linear(hidden_dim, out_f),)

#     def forward(self, x):
#         bs = x.size()[0]
#         x_feat = self.pool(x).view(bs, -1)
#         x = self.mlp(x_feat)
#         x = self.do(x)
#         return x, x_feat


# class MLP(nn.Module):
#     def __init__(self, in_f, hidden_dim, out_f):
#         super(MLP, self).__init__()
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.mlp = nn.Sequential(nn.Linear(in_f, hidden_dim),
#                                  nn.LeakyReLU(inplace=True),
#                                  nn.Linear(hidden_dim, hidden_dim),
#                                  nn.LeakyReLU(inplace=True),
#                                  nn.Linear(hidden_dim, out_f),)

#     def forward(self, x):
#         x = self.pool(x)
#         x = self.mlp(x)
#         return x

# class AdaIN(nn.Module):
#     def __init__(self, eps=1e-5):
#         super().__init__()
#         self.eps = eps
#         # self.l1 = nn.Linear(num_classes, in_channel*4, bias=True) #bias is good :)

#     def c_norm(self, x, bs, ch, eps=1e-7):
#         x_var = x.var(dim=-1) + eps
#         x_std = x_var.sqrt().view(bs, ch, 1, 1)
#         x_mean = x.mean(dim=-1).view(bs, ch, 1, 1)
#         return x_std, x_mean

#     def forward(self, x, y):
#         assert x.size(0) == y.size(0)
#         size = x.size()
#         bs, ch = size[:2]
#         x_ = x.view(bs, ch, -1)
#         y_ = y.reshape(bs, ch, -1)
#         x_std, x_mean = self.c_norm(x_, bs, ch, eps=self.eps)
#         y_std, y_mean = self.c_norm(y_, bs, ch, eps=self.eps)
#         out = ((x - x_mean.expand(size)) / x_std.expand(size)) \
#             * y_std.expand(size) + y_mean.expand(size)
#         return out

# class DecoderWithAdaIn(nn.Module):
#     def __init__(self, content_dim, fingerprint_dim):
#         super().__init__()
        
#         # MLP to map fingerprint → AdaIN parameters (γ, β)
#         self.style_mlp = nn.Sequential(
#             nn.Linear(fingerprint_dim, 256),
#             nn.ReLU(inplace=True),
#             nn.Linear(256, content_dim * 2)  # output γ and β
#         )

#         self.adain = AdaIN()

#         # A few upsampling + conv layers (reverse backbone)
#         self.up1 = nn.ConvTranspose2d(content_dim, 128, 4, stride=2, padding=1)
#         self.up2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
#         self.up3 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
#         self.final = nn.Conv2d(32, 3, kernel_size=3, padding=1)

#     def forward(self, content_feat, fingerprint_vec):
#         # get style parameters
#         style_params = self.style_mlp(fingerprint_vec)
        
#         # inject fingerprint into content via AdaIN
#         x = self.adain(content_feat, style_params)

#         # progressively upsample
#         x = F.relu(self.up1(x))
#         x = F.relu(self.up2(x))
#         x = F.relu(self.up3(x))
#         x = torch.sigmoid(self.final(x))  # reconstructed image in [0,1]

#         return x
    

# class DisentanglementEncoder(L.LightningModule):
#     """Encoder that disentangles features into content, specific, and common components"""
#     def __init__(self, input_channels=3, hidden_dim=512, num_classes=2, no_of_techniques=6):
#         super().__init__()
#         self.save_hyperparameters()

#         self.num_classes = num_classes

#         # Define base model
#         base_model = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)

#         # Encoders
#         self.encoder_content = nn.Sequential(*list(base_model.features.children()))
#         self.encoder_fingerprint = nn.Sequential(*list(base_model.features.children()))

#         self.encoder_feat_dim = base_model.classifier[1].in_features
#         self.half_fingerprint_dim = self.encoder_feat_dim // 2
#         self.decoder = DecoderWithAdaIn(content_dim=self.encoder_feat_dim,
#                                 fingerprint_dim=self.half_fingerprint_dim)

#         no_of_techniques = self.no_of_techniques
#         # Three separate heads for disentanglement
#         self.head_specific = Head(
#             in_f=self.half_fingerprint_dim,
#             hidden_dim=self.encoder_feat_dim,
#             out_f=no_of_techniques
#         )
#         self.head_fingerprint_common = Head(
#             in_f=self.half_fingerprint_dim,
#             hidden_dim=self.encoder_feat_dim,
#             out_f=self.num_classes
#         )
#         self.head_fused = Head(
#             in_f=self.half_fingerprint_dim,
#             hidden_dim=self.encoder_feat_dim,
#             out_f=self.num_classes
#         )

#         # Blocks
#         self.block_specific = Conv2d1x1(
#             in_f=self.encoder_feat_dim,
#             hidden_dim=self.half_fingerprint_dim,
#             out_f=self.half_fingerprint_dim
#         )
#         self.block_common = Conv2d1x1(
#             in_f=self.encoder_feat_dim,
#             hidden_dim=self.half_fingerprint_dim,
#             out_f=self.half_fingerprint_dim
#         )
#         self.block_fused = Conv2d1x1(
#             in_f=self.encoder_feat_dim,
#             hidden_dim=self.half_fingerprint_dim,
#             out_f=self.half_fingerprint_dim
#         )

#         self.adain = AdaIN()

#         # Loss functions
#         self.loss_func = {
#             "cls": nn.CrossEntropyLoss(),
#             "spe": nn.CrossEntropyLoss(),
#             "rec": nn.L1Loss(),
#             "con": ContrastiveLoss(),
#             "fuse": LDAMLoss(),
#         }

#         # Metrics
#         self.train_acc = BinaryAccuracy()
#         self.val_acc = BinaryAccuracy()
#         self.test_acc = BinaryAccuracy()

#         self.train_f1 = BinaryF1Score()
#         self.val_f1   = BinaryF1Score()
#         self.test_f1  = BinaryF1Score()

#         self.train_auc = BinaryAUROC()
#         self.val_auc   = BinaryAUROC()
#         self.test_auc  = BinaryAUROC()

#     def reconstruction_step(self, content_features, fingerprint_features):
#         """
#         content_features: (B, C, H, W)
#         fingerprint_features: (B, C, H, W)
#         """

#         # split batch into two halves for cross reconstruction
#         f1, f2 = fingerprint_features.chunk(2, dim=0)
#         c1, c2 = content_features.chunk(2, dim=0)

#         # self reconstructions
#         self_recon1 = self.decoder(c1, f1)
#         self_recon2 = self.decoder(c2, f2)

#         # cross reconstructions
#         cross_recon1 = self.decoder(c1, f2)
#         cross_recon2 = self.decoder(c2, f1)

#         return self_recon1, self_recon2, cross_recon1, cross_recon2

#     # def _step(self, batch, stage):
#     #     x, y = batch
#     #     feature_content = self.encoder_content(x)
#     #     feature_fingerprint = self.encoder_fingerprint(x)

#     #     # split the features into the specific and common forgery
#     #     fingerprint_specific = self.block_specific(feature_fingerprint)
#     #     fingerprint_common = self.block_common(feature_fingerprint)
#     #     fused_features = self.decoder(feature_content, fingerprint_common)

#     #     # classify using fused features during val/test
#     #     out_common, common_feat = self.head_fingerprint_common(fingerprint_common)
#     #     out_specific, specific_feat = self.head_specific(fingerprint_specific)
#     #     out_fused, fused_feat = self.head_fused(fused_features)

#     #     if stage != "train":

#     #         # probability_common = torch.softmax(out_common, dim=1)
#     #         probability_fused = torch.softmax(out_fused, dim=1)
#     #         _, preds = torch.max(out_fused, 1)

#     #         # Update metrics
#     #         acc  = getattr(self, f"{stage}_acc")
#     #         f1   = getattr(self, f"{stage}_f1")
#     #         auc  = getattr(self, f"{stage}_auc")

#     #         acc.update(preds, y.int())
#     #         f1.update(preds, y.int())
#     #         auc.update(preds, y.int())

#     #         # self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
#     #         self.log(f"{stage}_acc", acc, prog_bar=True, on_epoch=True)
#     #         self.log(f"{stage}_f1", f1, prog_bar=True, on_epoch=True)
#     #         self.log(f"{stage}_auc", auc, prog_bar=True, on_epoch=True)

#     #     f_all = torch.cat((fingerprint_specific, fingerprint_common), dim=1)

#     #     # self and cross reconstruct images using disentangled features
#     #     self_recon1, self_recon2, cross_recon1, cross_recon2 = self.reconstruction_step(feature_content, f_all)

#     #     # out_common, common_feat = self.head_fingerprint_common(fingerprint_common)
#     #     # out_specific, specific_feat = self.head_specific(fingerprint_specific)
#     #     # out_fused, fused_feat = self.head_fused(fused_features)

#     #     probability_common = torch.softmax(out_common, dim=1)
#     #     probability_specific = torch.softmax(out_specific, dim=1)
#     #     probability_fused = torch.softmax(out_fused, dim=1)

#     #     # get combined, real, fake imgs
#     #     cat_data = data_dict['image']

#     #     real_img, fake_img = cat_data.chunk(2, dim=0)
#     #     # get the reconstruction imgs
#     #     # cross_recon1, \
#     #     #     cross_recon2, \
#     #     #     self_recon1, \
#     #     #     self_recon2 \
#     #     #     = pred_dict['recontruction_imgs']
#     #     # get label
#     #     label = y if y == 0 else 1
#     #     label_spe = y
#     #     intersec_label = data_dict['intersec_label']

#     #     # get pred
#     #     pred = pred_dict['cls']
#     #     # print(pred, 'pred')
#     #     pred_spe = pred_dict['cls_spe']
#     #     pred_fair = pred_dict['cls_fair']

#     #     prob_fuse = pred_dict['prob_fused']
#     #     pred_fuse = pred_dict['cls_fused']

#     #     # 1. classification loss for domain-agnostic features
#     #     loss_sha = self.loss_func['cls'](pred, label)


#     #     # 2. classification loss for domain-specific features
#     #     loss_spe = self.loss_func['spe'](pred_spe, label_spe)

#     #     # 3. reconstruction loss
#     #     self_loss_reconstruction_1 = self.loss_func['rec'](
#     #         fake_img, self_recon1)
#     #     self_loss_reconstruction_2 = self.loss_func['rec'](
#     #         real_img, self_recon2)
#     #     cross_loss_reconstruction_1 = self.loss_func['rec'](
#     #         fake_img, cross_recon2)
#     #     cross_loss_reconstruction_2 = self.loss_func['rec'](
#     #         real_img, cross_recon1)
#     #     loss_reconstruction = \
#     #         self_loss_reconstruction_1 + self_loss_reconstruction_2 + \
#     #         cross_loss_reconstruction_1 + cross_loss_reconstruction_2

#     #     # 4. constrative loss
#     #     common_features = pred_dict['feat']
#     #     specific_features = pred_dict['feat_spe']
#     #     loss_con = self.loss_func['con'](
#     #         common_features, specific_features, label_spe)


#     #     # fused loss
#     #     outer_loss = []
#     #     inter_index = list(torch.unique(intersec_label))
#     #     loss_fuse_entropy = self.loss_func['fuse'](pred_fuse, label)
#     #     for index in inter_index:
#     #         ori_inter_loss = loss_fuse_entropy[intersec_label == index]
#     #         lamda_i_search_func = self.search_func(
#     #             ori_inter_loss, 0.9)
#     #         searched_lamda_i = optimize.fminbound(lamda_i_search_func, np.min(ori_inter_loss.cpu(
#     #         ).detach().numpy()) - 1000.0, np.max(ori_inter_loss.cpu().detach().numpy()))
#     #         inner_loss = self.searched_lamda_loss(
#     #             ori_inter_loss, searched_lamda_i, 0.9)
#     #         outer_loss.append(inner_loss)
#     #     outer_loss = torch.stack(outer_loss)
#     #     lamda_search_func = self.search_func_smooth(
#     #         outer_loss, 0.5, 0.001, 0.0001)

#     #     searched_lamda = optimize.fminbound(lamda_search_func, np.min(outer_loss.cpu(
#     #     ).detach().numpy()) - 1000.0, np.max(outer_loss.cpu().detach().numpy()))
#     #     loss_fuse = self.searched_lamda_loss_smooth(
#     #         outer_loss, searched_lamda, 0.5, 0.001, 0.0001)

#     #     # 6. total loss
#     #     loss = loss_sha + 0.1*loss_spe + 0.3 * \
#     #         loss_reconstruction + 0.05*loss_con + loss_fuse
#     #     loss_dict = {
#     #         'overall': loss,
#     #         'common': loss_sha,
#     #         'specific': loss_spe,
#     #         'reconstruction': loss_reconstruction,
#     #         'contrastive': loss_con,
#     #         'fusion': loss_fuse
#     #     }

#     #     return loss
#     def _step(self, batch, stage):
#         x = batch['image']
#         y = batch['label']

#         # encode
#         feature_content = self.encoder_content(x)
#         feature_fingerprint = self.encoder_fingerprint(x)

#         # disentangle
#         fingerprint_specific = self.block_specific(feature_fingerprint)
#         fingerprint_common = self.block_common(feature_fingerprint)

#         # fuse
#         fused_features = self.decoder(feature_content, fingerprint_common)

#         # classify
#         out_common, feat_common = self.head_fingerprint_common(fingerprint_common)
#         out_specific, feat_specific = self.head_specific(fingerprint_specific)
#         out_fused, feat_fused = self.head_fused(fused_features)

#         # reconstructions
#         f_all = torch.cat((fingerprint_specific, fingerprint_common), dim=1)
#         self_recon1, self_recon2, cross_recon1, cross_recon2 = self.reconstruction_step(feature_content, f_all)

#         # losses
#         loss_sha = self.loss_func['cls'](out_common, y)
#         loss_spe = self.loss_func['spe'](out_specific, y)
#         loss_recon = (
#             self.loss_func['rec'](x, self_recon1) +
#             self.loss_func['rec'](x, self_recon2) +
#             self.loss_func['rec'](x, cross_recon1) +
#             self.loss_func['rec'](x, cross_recon2)
#         )
#         loss_con = self.loss_func['con'](feat_common, feat_specific, y)
#         loss_fuse = self.loss_func['fuse'](out_fused, y)

#         loss = loss_sha + 0.1 * loss_spe + 0.3 * loss_recon + 0.05 * loss_con + loss_fuse

#         # metrics (for val/test only)
#         if stage != "train":
#             preds = torch.argmax(out_fused, dim=1)
#             acc = getattr(self, f"{stage}_acc")
#             f1 = getattr(self, f"{stage}_f1")
#             auc = getattr(self, f"{stage}_auc")
#             acc.update(preds, y.int())
#             f1.update(preds, y.int())
#             auc.update(preds, y.int())
#             self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
#             self.log(f"{stage}_acc", acc, prog_bar=True, on_epoch=True)
#             self.log(f"{stage}_f1", f1, prog_bar=True, on_epoch=True)
#             self.log(f"{stage}_auc", auc, prog_bar=True, on_epoch=True)

#         return loss

#     def training_step(self, batch, batch_idx):
#         return self._step(batch, "train")

#     def validation_step(self, batch, batch_idx):
#         return self._step(batch, "val")

#     def test_step(self, batch, batch_idx):
#         return self._step(batch, "test")

#     def predict_step(self, batch, batch_idx, dataloader_idx=0):
#         return self(batch)

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryAUROC

# -------------------
# Helper modules
# -------------------
class Conv2d1x1(nn.Module):
    def __init__(self, in_f, hidden_dim, out_f):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_f, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_f, kernel_size=1)
        )
    def forward(self, x):
        return self.net(x)

class Head(nn.Module):
    """Classification head with optional embedding output"""
    def __init__(self, in_f, hidden_dim, out_f):
        super().__init__()
        self.fc1 = nn.Linear(in_f, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_f)
    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)  # global pooling
        feat = F.relu(self.fc1(x))
        out = self.fc2(feat)
        return out, feat

class AdaIN(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
    def c_norm(self, x, bs, ch):
        var = x.var(dim=-1) + self.eps
        std = var.sqrt().view(bs, ch, 1, 1)
        mean = x.mean(dim=-1).view(bs, ch, 1, 1)
        return std, mean
    def forward(self, x, y):
        bs, ch = x.size(0), x.size(1)
        x_ = x.view(bs, ch, -1)
        y_ = y.view(bs, ch, -1)
        x_std, x_mean = self.c_norm(x_, bs, ch)
        y_std, y_mean = self.c_norm(y_, bs, ch)
        return ((x - x_mean) / x_std) * y_std + y_mean

class DecoderWithAdaIn(nn.Module):
    def __init__(self, content_dim, fingerprint_dim):
        super().__init__()
        self.adain = AdaIN()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(content_dim, 128, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()  # output normalized image
        )
    def forward(self, content, fingerprint_vec):
        # fingerprint_vec is (B, D), expand to (B, C, H, W)
        B, C, H, W = content.shape
        f_map = fingerprint_vec.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        fused = self.adain(content, f_map)
        return self.conv(fused)

# -------------------
# Contrastive Loss (pairwise within batch)
# -------------------
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    def forward(self, feat_common, feat_specific, labels):
        # Split into two halves
        f1, f2 = feat_common.chunk(2, 0)
        s1, s2 = feat_specific.chunk(2, 0)
        y1, y2 = labels.chunk(2, 0)

        # Distance between pairs
        dist = F.pairwise_distance(f1, f2, keepdim=True)

        # Positive if same label
        is_pos = (y1 == y2).float().unsqueeze(1)
        loss_pos = is_pos * torch.pow(dist, 2)
        loss_neg = (1 - is_pos) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        return torch.mean(loss_pos + loss_neg)

# -------------------
# Main Model
# -------------------
class DisentanglementEncoder(L.LightningModule):
    def __init__(self, num_classes=2, no_of_techniques=6, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        base_model = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
        self.encoder_content = nn.Sequential(*list(base_model.features.children()))
        self.encoder_fingerprint = nn.Sequential(*list(base_model.features.children()))
        self.encoder_feat_dim = base_model.classifier[1].in_features
        self.half_dim = self.encoder_feat_dim // 2

        self.decoder = DecoderWithAdaIn(content_dim=self.encoder_feat_dim,
                                        fingerprint_dim=self.half_dim)

        # Heads
        self.head_specific = Head(self.half_dim, self.encoder_feat_dim, no_of_techniques)
        self.head_common = Head(self.half_dim, self.encoder_feat_dim, num_classes)
        self.head_fused = Head(self.half_dim, self.encoder_feat_dim, num_classes)

        # Blocks
        self.block_specific = Conv2d1x1(self.encoder_feat_dim, self.half_dim, self.half_dim)
        self.block_common = Conv2d1x1(self.encoder_feat_dim, self.half_dim, self.half_dim)

        # Losses
        self.loss_func = {
            "cls": nn.CrossEntropyLoss(),
            "spe": nn.CrossEntropyLoss(),
            "rec": nn.L1Loss(),
            "con": ContrastiveLoss(),
            "fuse": nn.CrossEntropyLoss(),  # start simple
        }

        # Metrics
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()

        self.train_f1 = BinaryF1Score()
        self.val_f1   = BinaryF1Score()
        self.test_f1  = BinaryF1Score()

        self.train_auc = BinaryAUROC()
        self.val_auc   = BinaryAUROC()
        self.test_auc  = BinaryAUROC()

        self.lr = lr

    def reconstruction_step(self, content_features, fingerprint_features):
        f1, f2 = fingerprint_features.chunk(2, 0)
        c1, c2 = content_features.chunk(2, 0)
        self_recon1 = self.decoder(c1, f1.mean([2,3]))
        self_recon2 = self.decoder(c2, f2.mean([2,3]))
        cross_recon1 = self.decoder(c1, f2.mean([2,3]))
        cross_recon2 = self.decoder(c2, f1.mean([2,3]))
        return self_recon1, self_recon2, cross_recon1, cross_recon2

    def _step(self, batch, stage):
        x, y = batch["image"], batch["label"]

        # Encode
        feat_c = self.encoder_content(x)
        feat_f = self.encoder_fingerprint(x)

        # Disentangle
        f_spec = self.block_specific(feat_f)
        f_comm = self.block_common(feat_f)

        # Fuse
        fused = self.decoder(feat_c, f_comm.mean([2,3]))

        # Classify
        out_comm, emb_comm = self.head_common(f_comm)
        out_spec, emb_spec = self.head_specific(f_spec)
        out_fused, emb_fused = self.head_fused(fused)

        # Reconstructions
        f_all = torch.cat((f_spec, f_comm), dim=1)
        self_recon1, self_recon2, cross_recon1, cross_recon2 = self.reconstruction_step(feat_c, f_all)

        # Losses
        loss_sha = self.loss_func['cls'](out_comm, y)
        loss_spe = self.loss_func['spe'](out_spec, y)
        loss_recon = (
            self.loss_func['rec'](x, self_recon1) +
            self.loss_func['rec'](x, self_recon2) +
            self.loss_func['rec'](x, cross_recon1) +
            self.loss_func['rec'](x, cross_recon2)
        )
        loss_con = self.loss_func['con'](emb_comm, emb_spec, y)
        loss_fuse = self.loss_func['fuse'](out_fused, y)

        loss = loss_sha + 0.1*loss_spe + 0.3*loss_recon + 0.05*loss_con + loss_fuse

        # Metrics
        if stage != "train":
            preds = torch.argmax(out_fused, dim=1)
            acc = getattr(self, f"{stage}_acc")
            f1 = getattr(self, f"{stage}_f1")
            auc = getattr(self, f"{stage}_auc")
            acc.update(preds, y.int())
            f1.update(preds, y.int())
            auc.update(preds, y.int())
            self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
            self.log(f"{stage}_acc", acc, prog_bar=True, on_epoch=True)
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
        return torch.optim.Adam(self.parameters(), lr=self.lr)
