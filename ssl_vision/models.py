"""
Vision Transformer models for self-supervised learning (IBOT, DINO)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import timm
from typing import Optional


class DINOHead(nn.Module):
    """
    Projection head for DINO/DINOv2
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_bn: bool = False,
        norm_last_layer: bool = True,
        nlayers: int = 3,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)

        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())

            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())

            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)

        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class IBOTHead(nn.Module):
    """
    Projection head for IBOT (patch-level predictions)
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        patch_out_dim: int,
        norm_last_layer: bool = False,
        shared_head: bool = True,
    ):
        super().__init__()
        self.shared_head = shared_head

        # Head for [CLS] token
        self.cls_head = DINOHead(
            in_dim=in_dim,
            out_dim=out_dim,
            norm_last_layer=norm_last_layer,
        )

        # Head for patch tokens
        if shared_head:
            self.patch_head = self.cls_head
        else:
            self.patch_head = DINOHead(
                in_dim=in_dim,
                out_dim=patch_out_dim,
                norm_last_layer=norm_last_layer,
            )

    def forward(self, x):
        # x: [B, N, D] where N = 1 + num_patches
        cls_token = x[:, 0]
        patch_tokens = x[:, 1:]

        cls_output = self.cls_head(cls_token)
        patch_output = self.patch_head(patch_tokens)

        return cls_output, patch_output


class MultiCropWrapper(nn.Module):
    """
    Wrapper to handle multiple crops for self-supervised learning
    """
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x, return_backbone_feat=False, masked_indices=None):
        # Handle multi-crop input
        if not isinstance(x, list):
            x = [x]

        # Debug output (only once)
        if not hasattr(MultiCropWrapper, '_debug_printed'):
            print(f"\n[DEBUG MULTICROP] Input: {len(x)} crops")
            print(f"[DEBUG MULTICROP] Crop shapes: {[crop.shape for crop in x]}")
            MultiCropWrapper._debug_printed = True

        # Group crops by resolution for efficient processing
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True
            )[1], 0
        )

        if not hasattr(MultiCropWrapper, '_debug_printed2'):
            print(f"[DEBUG MULTICROP] Resolution groups: {idx_crops.tolist()}")
            MultiCropWrapper._debug_printed2 = True

        start_idx = 0
        output = []
        for end_idx in idx_crops:
            # Get the crops for this resolution group
            crops_in_group = x[start_idx:end_idx]
            num_crops_in_group = len(crops_in_group)

            # Concatenate crops in this group
            _out = self.backbone(torch.cat(crops_in_group))

            if return_backbone_feat:
                # Split output back into individual crops
                # _out shape: [batch_size * num_crops, num_tokens, embed_dim]
                batch_size = crops_in_group[0].shape[0]
                _out_split = _out.reshape(num_crops_in_group, batch_size, *_out.shape[1:])
                for i in range(num_crops_in_group):
                    output.append(_out_split[i])
            else:
                # Pass through head
                _out = self.head(_out)

                # Split output back into individual crops
                # For DINO: _out is a tensor [batch_size * num_crops, out_dim]
                # For IBOT: _out is a tuple of tensors
                if isinstance(_out, tuple):
                    # IBOT case: (cls_output, patch_output)
                    batch_size = crops_in_group[0].shape[0]
                    cls_out, patch_out = _out

                    # Split cls_output: [batch_size * num_crops, out_dim]
                    cls_out_split = cls_out.reshape(num_crops_in_group, batch_size, -1)
                    # Split patch_output: [batch_size * num_crops, num_patches, patch_out_dim]
                    patch_out_split = patch_out.reshape(num_crops_in_group, batch_size, *patch_out.shape[1:])

                    for i in range(num_crops_in_group):
                        output.append((cls_out_split[i], patch_out_split[i]))
                else:
                    # DINO case: single tensor [batch_size * num_crops, out_dim]
                    batch_size = crops_in_group[0].shape[0]
                    _out_split = _out.reshape(num_crops_in_group, batch_size, -1)
                    for i in range(num_crops_in_group):
                        output.append(_out_split[i])

            start_idx = end_idx

        if not hasattr(MultiCropWrapper, '_debug_printed3'):
            print(f"[DEBUG MULTICROP] Output: {len(output)} crops")
            if len(output) > 0:
                if isinstance(output[0], tuple):
                    print(f"[DEBUG MULTICROP] First output (IBOT): cls shape {output[0][0].shape}, patch shape {output[0][1].shape}")
                else:
                    print(f"[DEBUG MULTICROP] First output (DINO) shape: {output[0].shape}")
            MultiCropWrapper._debug_printed3 = True

        return output


def create_vision_transformer(cfg):
    """
    Create Vision Transformer backbone using timm
    """
    vit_cfg = cfg.model.vit

    # Use image size from data config if available, otherwise from model config
    img_size = cfg.data.image_size if hasattr(cfg.data, 'image_size') else vit_cfg.img_size

    # Create model using timm
    model = timm.create_model(
        cfg.model.architecture,
        pretrained=False,
        num_classes=0,  # Remove classification head
        img_size=img_size,
        patch_size=vit_cfg.patch_size,
        embed_dim=vit_cfg.embed_dim,
        depth=vit_cfg.depth,
        num_heads=vit_cfg.num_heads,
        mlp_ratio=vit_cfg.mlp_ratio,
        drop_path_rate=vit_cfg.drop_path_rate,
    )

    # Modify to return all tokens (CLS + patches)
    original_forward = model.forward_features

    def new_forward(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x  # Return all tokens

    model.forward_features = partial(new_forward, model)
    model.forward = model.forward_features

    return model


def create_ibot_model(cfg):
    """
    Create IBOT model (student and teacher)
    """
    # Create backbone
    backbone = create_vision_transformer(cfg)
    embed_dim = cfg.model.vit.embed_dim

    # Create head
    head = IBOTHead(
        in_dim=embed_dim,
        out_dim=cfg.model.ibot.out_dim,
        patch_out_dim=cfg.model.ibot.patch_out_dim,
        norm_last_layer=cfg.model.ibot.norm_last_layer,
    )

    # Wrap with multi-crop wrapper
    student = MultiCropWrapper(backbone, head)
    teacher = MultiCropWrapper(
        create_vision_transformer(cfg),
        IBOTHead(
            in_dim=embed_dim,
            out_dim=cfg.model.ibot.out_dim,
            patch_out_dim=cfg.model.ibot.patch_out_dim,
            norm_last_layer=cfg.model.ibot.norm_last_layer,
        )
    )

    # Teacher starts with same weights as student
    teacher.load_state_dict(student.state_dict())

    # Freeze teacher
    for p in teacher.parameters():
        p.requires_grad = False

    return student, teacher


def create_dino_model(cfg):
    """
    Create DINO/DINOv2/DINOv3 model (student and teacher)
    """
    # Create backbone
    backbone = create_vision_transformer(cfg)
    embed_dim = cfg.model.vit.embed_dim

    # Determine bottleneck_dim
    bottleneck_dim = cfg.model.dino.get('bottleneck_dim', 256)

    # Create head (only for CLS token in pure DINO)
    head = DINOHead(
        in_dim=embed_dim,
        out_dim=cfg.model.dino.out_dim,
        norm_last_layer=cfg.model.dino.norm_last_layer,
        bottleneck_dim=bottleneck_dim,
    )

    # Wrap with multi-crop wrapper
    student = MultiCropWrapper(backbone, head)
    teacher = MultiCropWrapper(
        create_vision_transformer(cfg),
        DINOHead(
            in_dim=embed_dim,
            out_dim=cfg.model.dino.out_dim,
            norm_last_layer=cfg.model.dino.norm_last_layer,
            bottleneck_dim=bottleneck_dim,
        )
    )

    # Teacher starts with same weights as student
    teacher.load_state_dict(student.state_dict())

    # Freeze teacher
    for p in teacher.parameters():
        p.requires_grad = False

    return student, teacher


@torch.no_grad()
def update_teacher(student, teacher, momentum):
    """
    Update teacher with exponential moving average of student weights
    """
    for param_s, param_t in zip(student.parameters(), teacher.parameters()):
        param_t.data.mul_(momentum).add_((1 - momentum) * param_s.detach().data)


class DINOLoss(nn.Module):
    """
    DINO loss with temperature scaling and centering
    """
    def __init__(
        self,
        out_dim: int,
        ncrops: int = 2,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.04,
        warmup_teacher_temp_epochs: int = 0,
        nepochs: int = 100,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))

        # Temperature schedule
        self.teacher_temp_schedule = torch.cat([
            torch.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            torch.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ])

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        # student_output and teacher_output are lists of tensors (one per crop)
        # For IBOT models, outputs are tuples (cls_output, patch_output), so extract cls_output
        # For DINO models, outputs are tensors directly

        # Get teacher temperature for this epoch
        temp = self.teacher_temp_schedule[epoch]

        # Process student outputs
        student_out = []
        for s in student_output:
            if isinstance(s, tuple):
                student_out.append(s[0] / self.student_temp)  # Use CLS token output
            else:
                student_out.append(s / self.student_temp)

        # Process teacher outputs with centering and sharpening
        teacher_out = []
        for t in teacher_output:
            if isinstance(t, tuple):
                t = t[0]  # Use CLS token output
            teacher_out.append(F.softmax((t - self.center) / temp, dim=-1).detach())

        # Ensure we have outputs to compute loss
        if len(student_out) == 0 or len(teacher_out) == 0:
            raise ValueError(f"Empty outputs: student_out={len(student_out)}, teacher_out={len(teacher_out)}")

        # For multi-crop, we need at least 2 crops to compute meaningful loss
        if len(student_out) < 2:
            raise ValueError(f"Need at least 2 crops for multi-crop training, got {len(student_out)} student crops")

        total_loss = 0
        n_loss_terms = 0

        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # Skip same crop
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

        total_loss /= n_loss_terms

        # Update center
        self.update_center(teacher_output)

        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output
        """
        # Extract CLS token outputs if using IBOT (tuple outputs)
        teacher_tensors = []
        for t in teacher_output:
            if isinstance(t, tuple):
                teacher_tensors.append(t[0])  # Use CLS token output
            else:
                teacher_tensors.append(t)

        batch_center = torch.cat(teacher_tensors).mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)




