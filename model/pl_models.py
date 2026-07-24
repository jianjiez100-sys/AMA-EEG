import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, CohenKappa

# Ensure these imports work based on your directory structure
from .loss.con_loss import MultiModalInfoNCELoss, MultiModalSupConLoss, CrossModalSupConLoss, VideoSupConLoss, SemanticVideoSupConLoss
from .metric.metrics import accuracy
from .models import ResidualAdd


# =========================================================================
# Helper Functions
# =========================================================================
def compute_entropy(logits):
    """
    Computes entropy of classification predictions.
    Entropy = - sum(p * log(p))
    """
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    entropy = -(probs * log_probs).sum(dim=1, keepdim=True)  # (Batch, 1)
    return entropy


# =========================================================================
# Lightning Module: ExtractorModel
# =========================================================================
class ExtractorModel(pl.LightningModule):
    def __init__(self, model, cfg) -> None:
        super().__init__()
        # Ignore model saving to reduce ckpt size
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.cfg = cfg
        self.lr = cfg.lr
        self.wd = cfg.wd
        self.max_epochs = cfg.max_epochs

        self.restart_times = getattr(cfg, 'restart_times', 1)
        self.w_clip = getattr(cfg, 'w_clip', 1.0)
        # 0:Text, 1:Image, 2:Dynamic, 3:Static
        self.pretrain_mode = int(cfg.pretrain_mode) if 'pretrain_mode' in cfg else 2
        self.n_class = getattr(cfg, 'n_class', 9)

        self.image_feat_dim = getattr(cfg, 'image_feat_dim', 1024)
        if self.image_feat_dim != 1024:
            self.image_input_proj = nn.Linear(self.image_feat_dim, 1024, bias=False)

        # 情感类别名（用于日志）
        if self.n_class == 2:
            self.emotion_names = ["0_Negative", "1_Positive"]
        else:
            self.emotion_names = [
                "0_Anger", "1_Disgust", "2_Fear", "3_Sadness",
                "4_Neutral", "5_Amusement", "6_Inspiration", "7_Joy", "8_Tenderness"
            ]

        # Text / Image 投影头（可学习，将预提取特征映射到与 EEG 对齐的公共空间）
        # self.text_projector = nn.Sequential(
        #     nn.Linear(1024, 2048, bias=False),
        #     # nn.LayerNorm(2048),
        #     # nn.Dropout(0.2),
        #     nn.BatchNorm1d(2048),
        #     nn.Dropout(0.2),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(2048, 1024, bias=True)
        # )
        # self.image_projector = nn.Sequential(
        #     nn.Linear(1024, 2048, bias=False),
        #     nn.BatchNorm1d(2048),
        #     nn.Dropout(0.2),
        #     # nn.LayerNorm(2048),
        #     # nn.Dropout(0.2),  # <- 给严师降降智，保护差生
        #     nn.ReLU(inplace=True),
        #     nn.Linear(2048, 1024, bias=True)
        # )
        # --- Text 投影头：独立定义，3 层瓶颈结构 (1024-512-512-1024) ---
        # self.text_projector = nn.Sequential(
        #     # 第一层：降维压缩，提取核心特征
        #     nn.Linear(1024, 512, bias=False),
        #     nn.BatchNorm1d(512),
        #     nn.GELU(),
        #     nn.Dropout(0.2),
        #     # 第二层：深层非线性重构
        #     nn.Linear(512, 512, bias=False),
        #     nn.BatchNorm1d(512),
        #     nn.GELU(),
        #     nn.Dropout(0.2),
        #     # 第三层：映射到对比空间
        #     nn.Linear(512, 1024, bias=True)
        # )
        #
        # # --- Image 投影头：独立定义，结构对称，参数独立 ---
        # self.image_projector = nn.Sequential(
        #     # 第一层：降维压缩
        #     nn.Linear(1024, 512, bias=False),
        #     nn.BatchNorm1d(512),
        #     nn.GELU(),
        #     nn.Dropout(0.2),
        #     # 第二层：深层非线性重构
        #     nn.Linear(512, 512, bias=False),
        #     nn.BatchNorm1d(512),
        #     nn.GELU(),
        #     nn.Dropout(0.2),
        #     # 第三层：映射到对比空间
        #     nn.Linear(512, 1024, bias=True)
        # )
        # --- Text / Image 投影头 (始终 residual + LayerNorm) ---
        use_modal_proj = getattr(cfg, 'use_modal_proj', False)
        self._need_projectors = (self.pretrain_mode in [2, 3]) or use_modal_proj
        self._use_modal_proj = use_modal_proj

        self._has_align = getattr(cfg, 'use_two_stage_projector', False)

        def _make_proj():
            return nn.Sequential(
                nn.Linear(1024, 1024),
                ResidualAdd(nn.Sequential(
                    nn.GELU(),
                    nn.Linear(1024, 1024),
                    nn.Dropout(0.2),
                )),
                nn.LayerNorm(1024),
            )

        if self._need_projectors:
            # ======== Align 层 (独立前置, 冻结, fp32) ========
            if self._has_align:
                # 保存 RNG → 创建 Align → 恢复 RNG → 创建 Projector
                # 确保 Projector 初始化与原版项目完全一致
                rng_state = torch.get_rng_state()
                self.text_align_proj = _make_proj()
                self.image_align_proj = _make_proj()
                torch.set_rng_state(rng_state)

                # 加载 fusion8 预训练权重
                pretrained_text = getattr(cfg, 'pretrained_text_proj', '')
                pretrained_image = getattr(cfg, 'pretrained_image_proj', '')
                if pretrained_text:
                    state = torch.load(pretrained_text, map_location='cpu', weights_only=True)
                    if any(k.startswith('net.') for k in state.keys()):
                        state = {k.replace('net.', ''): v for k, v in state.items()}
                    self.text_align_proj.load_state_dict(state, strict=True)
                    print(f"📦 [Align] Loaded text_align from {pretrained_text}")
                if pretrained_image:
                    state = torch.load(pretrained_image, map_location='cpu', weights_only=True)
                    if any(k.startswith('net.') for k in state.keys()):
                        state = {k.replace('net.', ''): v for k, v in state.items()}
                    self.image_align_proj.load_state_dict(state, strict=True)
                    print(f"📦 [Align] Loaded image_align from {pretrained_image}")

                # 永久 eval + fp32 + 冻结
                self.text_align_proj.eval()
                self.image_align_proj.eval()
                # 锁定 training 状态 (必须在 eval() 之后)
                self.text_align_proj.train = lambda mode=True: self.text_align_proj
                self.image_align_proj.train = lambda mode=True: self.image_align_proj
                # fp32 前向包装
                def _wrap_fp32(mod):
                    orig = mod.forward
                    def fp32_fwd(x):
                        with torch.amp.autocast('cuda', enabled=False):
                            return orig(x.float())
                    mod.forward = fp32_fwd
                _wrap_fp32(self.text_align_proj)
                _wrap_fp32(self.image_align_proj)
                # 冻结参数
                for p in self.text_align_proj.parameters():
                    p.requires_grad = False
                for p in self.image_align_proj.parameters():
                    p.requires_grad = False
                print(f"🔒 Align frozen, fp32, eval mode")

            # ======== Projector (可训练, 随机初始化, 与原版一致) ========
            self.text_projector = _make_proj()
            self.image_projector = _make_proj()

        # ======== Mode 初始化 ========
        if self.pretrain_mode == 2:
            self.n_class = getattr(cfg, 'n_class', 9)
            self.distill_criterion = MultiModalInfoNCELoss(temperature=cfg.loss_temp, threshold=cfg.loss_threshold, mask_mode=cfg.mask_mode)
            self.conf_temp = getattr(cfg, 'conf_temp', 0.07)
            feat_dim = 1024
            self.text_probe = nn.Linear(feat_dim, self.n_class)
            self.image_probe = nn.Linear(feat_dim, self.n_class)
            self.probe_criterion = nn.CrossEntropyLoss()
            self.probe_loss_weight = getattr(cfg, 'probe_loss_weight', 1.0)
        elif self.pretrain_mode == 3:
            self.fusion_alpha = getattr(cfg, 'fusion_alpha', 0.5)
            print(f"🔒 Static Fusion α={self.fusion_alpha}")
            self.distill_criterion = MultiModalInfoNCELoss(temperature=cfg.loss_temp, threshold=cfg.loss_threshold, mask_mode=cfg.mask_mode)
        elif self.pretrain_mode == 0:
            self.criterion = MultiModalInfoNCELoss(temperature=cfg.loss_temp, threshold=cfg.loss_threshold, mask_mode=cfg.mask_mode)
        else:
            self.criterion = MultiModalInfoNCELoss(temperature=cfg.loss_temp, threshold=cfg.loss_threshold, mask_mode=cfg.mask_mode)

    def forward(self, x):
        # Turn on saveFea during inference/finetuning
        if hasattr(self.model, 'set_saveFea'):
            self.model.set_saveFea(True)
        return self.model(x)

    def set_return_layer(self, layer):
        if hasattr(self.model, 'set_return_layer'):
            self.model.set_return_layer(layer)

    def set_return_pre_tconv(self, flag):
        if hasattr(self.model, 'set_return_pre_tconv'):
            self.model.set_return_pre_tconv(flag)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=self.max_epochs // self.restart_times, eta_min=0, last_epoch=-1
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, batch, batch_idx):
        # 1. 解包数据
        eeg, labels, txt_feat, img_feat, *extras = batch
        vid_ids = extras[0].view(-1) if extras else None
        sub_ids = extras[1].view(-1) if len(extras) > 1 else None

        # 2. 类型转换与维度处理
        eeg = eeg.float()
        txt_feat = txt_feat.float()
        img_feat = img_feat.float()
        labels = labels.long()

        # ===== 二分类预训练：剔除中性样本 =====
        n_classes = getattr(self, 'n_class', 9)
        if n_classes == 2:
            not_neutral = (labels != 4)  # label 4 = 中性
            eeg = eeg[not_neutral]
            txt_feat = txt_feat[not_neutral]
            img_feat = img_feat[not_neutral]
            labels = labels[not_neutral]
            # 重映射: 负类(0-3)→0, 正类(5-8)→1
            labels = (labels >= 5).long()
            if vid_ids is not None:
                vid_ids = vid_ids[not_neutral]
            if sub_ids is not None:
                sub_ids = sub_ids[not_neutral]

        # 如果特征维度是 (Batch, Seq, Dim)，取平均变为 (Batch, Dim)
        if txt_feat.ndim == 3: txt_feat = txt_feat.mean(dim=1)
        if img_feat.ndim == 3: img_feat = img_feat.mean(dim=1)

        # 可学习线性投影: 1664 → 1024 (替代 PCA)
        if hasattr(self, 'image_input_proj'):
            img_feat = self.image_input_proj(img_feat)

        # 注意：此处不做归一化。
        # Mode 0/1: MultiModalInfoNCELoss.forward 内部统一归一化
        # Mode 2/3: 各分支内部的 F.normalize(t_feat/i_feat) 负责归一化

        # 确保 Backbone 不处于”只输出特征”的模式
        if hasattr(self.model, 'set_saveFea'):
            self.model.set_saveFea(False)

        loss = 0.0
        mode_name = ""

        # 初始化日志字典
        log_data = {
            'ext/train/lr': self.optimizers().param_groups[-1]['lr']
        }

        # ================= 模式逻辑分支 =================

        if self.pretrain_mode == 2:
            mode_name = "fusion_dynamic"

            # === Step 1: Align 前置投影 (冻结, 无梯度) ===
            # 保存 raw 特征给探针用
            txt_feat_raw = txt_feat
            img_feat_raw = img_feat
            if self._has_align:
                with torch.no_grad():
                    txt_feat = self.text_align_proj(txt_feat)  # raw → projected
                    img_feat = self.image_align_proj(img_feat)

            # === Step 2: Projector 用投影特征, Probe 用原始特征 ===
            t_feat = F.normalize(txt_feat.detach(), dim=1)
            i_feat = F.normalize(img_feat.detach(), dim=1)
            t_proj = F.normalize(self.text_projector(t_feat), dim=1)
            i_proj = F.normalize(self.image_projector(i_feat), dim=1)

            # 探针在原始 CLIP 特征上做分类
            t_feat_probe = F.normalize(txt_feat_raw.detach(), dim=1)
            i_feat_probe = F.normalize(img_feat_raw.detach(), dim=1)
            t_logits = self.text_probe(t_feat_probe)
            i_logits = self.image_probe(i_feat_probe)
            loss_probe_t = self.probe_criterion(t_logits, labels)
            loss_probe_i = self.probe_criterion(i_logits, labels)
            loss_probes = loss_probe_t + loss_probe_i

            with torch.no_grad():
                entropy_t = compute_entropy(t_logits)
                entropy_i = compute_entropy(i_logits)
                diff = (entropy_i - entropy_t) / self.conf_temp
                inst_alpha = torch.sigmoid(diff)

                alpha_txt = torch.zeros_like(inst_alpha)
                for vid in torch.unique(vid_ids) if vid_ids is not None else torch.unique(labels):
                    mask = (vid_ids == vid) if vid_ids is not None else (labels == vid)
                    alpha_txt[mask] = inst_alpha[mask].mean()

                for lbl in torch.unique(labels):
                    label_idx = lbl.item()
                    if 0 <= label_idx < len(self.emotion_names):
                        log_data[f'alpha_class/{self.emotion_names[label_idx]}'] = inst_alpha[labels == lbl].mean()

            fused_target = alpha_txt * t_proj + (1.0 - alpha_txt) * i_proj
            fused_target = F.normalize(fused_target, dim=1)

            proj_eeg = self.model(eeg)
            loss_distill = self.distill_criterion(proj_eeg, fused_target, original_targets=txt_feat, vid_ids=vid_ids, sub_ids=sub_ids)
            loss = loss_distill + self.probe_loss_weight * loss_probes
            log_data['ext/train/loss_fusion'] = loss_distill
            log_data['ext/train/loss_probes'] = loss_probes
            log_data['ext/train/alpha_mean'] = alpha_txt.mean()

        elif self.pretrain_mode == 3:
            # === Mode 3: Static Fusion (0.5/0.5) ===
            mode_name = "fusion_static"
            if self._has_align:
                with torch.no_grad():
                    txt_feat = self.text_align_proj(txt_feat)
                    img_feat = self.image_align_proj(img_feat)
            t_feat = F.normalize(txt_feat, dim=1)
            i_feat = F.normalize(img_feat, dim=1)
            t_proj = F.normalize(self.text_projector(t_feat), dim=1)
            i_proj = F.normalize(self.image_projector(i_feat), dim=1)
            fused_target = F.normalize(self.fusion_alpha * t_proj + (1 - self.fusion_alpha) * i_proj, dim=1)
            proj_eeg = self.model(eeg)
            loss = self.distill_criterion(proj_eeg, fused_target, original_targets=txt_feat, vid_ids=vid_ids, sub_ids=sub_ids)

        elif self.pretrain_mode == 0:
            mode_name = "text"
            if self._has_align:
                with torch.no_grad():
                    txt_feat = self.text_align_proj(txt_feat)
            if self._use_modal_proj:
                target_feat = self.text_projector(txt_feat)
            else:
                target_feat = txt_feat
            proj_eeg = self.model(eeg)
            loss = self.criterion(proj_eeg, target_feat, vid_ids=vid_ids, sub_ids=sub_ids)

        elif self.pretrain_mode == 1:
            mode_name = "image"
            if self._has_align:
                with torch.no_grad():
                    img_feat = self.image_align_proj(img_feat)
            if self._use_modal_proj:
                target_feat = self.image_projector(img_feat)
            else:
                target_feat = img_feat
            proj_eeg = self.model(eeg)
            loss = self.criterion(proj_eeg, target_feat, vid_ids=vid_ids, sub_ids=sub_ids)

        # ==============================================

        # 应用 Loss 权重系数
        total_loss = self.w_clip * loss

        # 更新总 Loss 到日志
        log_data['ext/train/loss'] = total_loss
        if mode_name:
            log_data[f'ext/train/loss_{mode_name}'] = loss

        # 统一提交日志
        self.log_dict(log_data, on_step=False, on_epoch=True, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        # 1. 解包与数据准备
        eeg, labels, txt_feat, img_feat, *extras = batch
        vid_ids = extras[0].view(-1) if extras else None
        sub_ids = extras[1].view(-1) if len(extras) > 1 else None

        eeg = eeg.float()
        txt_feat = txt_feat.float()
        img_feat = img_feat.float()
        # 这里的 labels 仅仅是为了算 Probe Loss 评估用的，绝对不进对齐 Loss
        labels = labels.long()

        # ===== 二分类预训练：剔除中性样本（与 training_step 保持一致）=====
        n_classes = getattr(self, 'n_class', 9)
        if n_classes == 2:
            not_neutral = (labels != 4)
            eeg = eeg[not_neutral]
            txt_feat = txt_feat[not_neutral]
            img_feat = img_feat[not_neutral]
            labels = labels[not_neutral]
            labels = (labels >= 5).long()
            if vid_ids is not None:
                vid_ids = vid_ids[not_neutral]
            if sub_ids is not None:
                sub_ids = sub_ids[not_neutral]

        if txt_feat.ndim == 3: txt_feat = txt_feat.mean(dim=1)
        txt_feat = F.normalize(txt_feat, dim=1)
        if img_feat.ndim == 3: img_feat = img_feat.mean(dim=1)
        if hasattr(self, 'image_input_proj'):
            img_feat = self.image_input_proj(img_feat)
        img_feat = F.normalize(img_feat, dim=1)

        if hasattr(self.model, 'set_saveFea'):
            self.model.set_saveFea(False)

        loss = 0.0
        mode_name = ""
        target_feat = None  # 用于统一记录要对齐的目标特征

        # ================= 模式逻辑分支 =================
        if self.pretrain_mode == 2:
            mode_name = "fusion_dynamic"

            # === Step 1: Align 前置投影 ===
            txt_feat_raw = txt_feat
            img_feat_raw = img_feat
            if self._has_align:
                with torch.no_grad():
                    txt_feat = self.text_align_proj(txt_feat)
                    img_feat = self.image_align_proj(img_feat)

            # === Step 2: Projector 用投影特征, Probe 用原始特征 ===
            t_feat = F.normalize(txt_feat.detach(), dim=1)
            i_feat = F.normalize(img_feat.detach(), dim=1)
            t_proj = F.normalize(self.text_projector(t_feat), dim=1)
            i_proj = F.normalize(self.image_projector(i_feat), dim=1)

            t_feat_probe = F.normalize(txt_feat_raw.detach(), dim=1)
            i_feat_probe = F.normalize(img_feat_raw.detach(), dim=1)
            t_logits = self.text_probe(t_feat_probe)
            i_logits = self.image_probe(i_feat_probe)
            self.log('val/probe_txt_loss', self.probe_criterion(t_logits, labels), prog_bar=False)
            self.log('val/probe_img_loss', self.probe_criterion(i_logits, labels), prog_bar=False)

            with torch.no_grad():
                entropy_t = compute_entropy(t_logits)
                entropy_i = compute_entropy(i_logits)
                diff = (entropy_i - entropy_t) / self.conf_temp
                alpha_txt = torch.sigmoid(diff)

            fused_target = alpha_txt * t_proj + (1.0 - alpha_txt) * i_proj
            target_feat = F.normalize(fused_target, dim=1)
            proj_eeg = self.model(eeg)

            loss = self.distill_criterion(proj_eeg, target_feat, original_targets=txt_feat, vid_ids=vid_ids, sub_ids=sub_ids)

        elif self.pretrain_mode == 3:
            mode_name = "fusion_static"
            if self._has_align:
                with torch.no_grad():
                    txt_feat = self.text_align_proj(txt_feat)
                    img_feat = self.image_align_proj(img_feat)
            t_feat = F.normalize(txt_feat, dim=1)
            i_feat = F.normalize(img_feat, dim=1)
            t_proj = F.normalize(self.text_projector(t_feat), dim=1)
            i_proj = F.normalize(self.image_projector(i_feat), dim=1)
            target_feat = F.normalize(self.fusion_alpha * t_proj + (1 - self.fusion_alpha) * i_proj, dim=1)
            proj_eeg = self.model(eeg)
            loss = self.distill_criterion(proj_eeg, target_feat, original_targets=txt_feat, vid_ids=vid_ids, sub_ids=sub_ids)

        elif self.pretrain_mode == 0:
            mode_name = "text"
            if self._has_align:
                with torch.no_grad():
                    txt_feat = self.text_align_proj(txt_feat)
            if self._use_modal_proj:
                target_feat = F.normalize(self.text_projector(txt_feat), dim=1)
            else:
                target_feat = F.normalize(txt_feat, dim=1)
            proj_eeg = self.model(eeg)
            loss = self.criterion(proj_eeg, target_feat, vid_ids=vid_ids, sub_ids=sub_ids)

        elif self.pretrain_mode == 1:
            mode_name = "image"
            if self._has_align:
                with torch.no_grad():
                    img_feat = self.image_align_proj(img_feat)
            if self._use_modal_proj:
                target_feat = F.normalize(self.image_projector(img_feat), dim=1)
            else:
                target_feat = F.normalize(img_feat, dim=1)
            proj_eeg = self.model(eeg)
            loss = self.criterion(proj_eeg, target_feat, vid_ids=vid_ids, sub_ids=sub_ids)

        total_loss = self.w_clip * loss

        # =========================================================
        # 🚀 [新增监控指标：跨模态对齐效果体检]
        # =========================================================
        if target_feat is not None and proj_eeg is not None:
            # 1. 计算当前批次的余弦相似度矩阵
            sim_matrix = torch.matmul(proj_eeg, target_feat.T)
            batch_size = sim_matrix.shape[0]

            # 2. 计算跨模态检索准确率 (Retrieval Acc)
            # SupCon 语义：找到同情感的样本即为正确，不要求 index 对齐
            e2t_retrieved = labels[sim_matrix.argmax(dim=1)]   # EEG_i 最相似的 Text 的标签
            e2t_acc = (e2t_retrieved == labels).float().mean()

            t2e_retrieved = labels[sim_matrix.argmax(dim=0)]   # Text_j 最相似的 EEG 的标签
            t2e_acc = (t2e_retrieved == labels).float().mean()

            # 3. 计算正负样本相似度鸿沟 (Similarity Gap)
            pos_sim = torch.diag(sim_matrix).mean()
            mask = torch.eye(batch_size, dtype=torch.bool, device=self.device)
            if batch_size > 1:
                neg_sim = sim_matrix[~mask].mean()
            else:
                neg_sim = torch.tensor(0.0, device=self.device)

            sim_gap = pos_sim - neg_sim

            # 记录到 TensorBoard
            self.log('val/E2T_Acc', e2t_acc, prog_bar=True)
            self.log('val/T2E_Acc', t2e_acc, prog_bar=False)
            self.log('val/Pos_Sim', pos_sim, prog_bar=False)
            self.log('val/Neg_Sim', neg_sim, prog_bar=False)
            self.log('val/Sim_Gap', sim_gap, prog_bar=True)

            # =====================================================
            # 🌟 [新增]：零样本 9 分类情感准确率 (Zero-Shot Accuracy)
            # =====================================================
            # 逻辑：脑电找到最相似的目标特征 (文本/融合)，看看那个目标特征属于什么情感
            best_match_indices = sim_matrix.argmax(dim=1)
            pred_labels = labels[best_match_indices]
            zero_shot_acc = (pred_labels == labels).float().mean()

            # 记录到进度条和 TensorBoard，绝对不参与 loss 计算！
            self.log('val/ZeroShot_Acc', zero_shot_acc, prog_bar=True)
        # =========================================================

        # =====================================================
        # 🌟 [新增] Backbone 原型分类 (Prototype Accuracy)
        # 用 backbone 原始特征做最近邻分类，直接衡量特征空间类别可分性
        # =====================================================
        EMOTION_NAMES = ['Anger', 'Disgust', 'Fear', 'Sadness', 'Neutral',
                         'Amusement', 'Inspiration', 'Joy', 'Tenderness']

        self.model.set_saveFea(True)
        with torch.no_grad():
            raw_eeg = self.model(eeg)
        self.model.set_saveFea(False)

        unique_labels = torch.unique(labels)
        if len(unique_labels) >= 2:
            proto_keys = sorted(unique_labels.tolist())
            # 计算各类质心
            centroids = {}
            counts = {}
            for k in proto_keys:
                mask = (labels == k)
                counts[k] = mask.sum()
                centroids[k] = raw_eeg[mask].mean(dim=0)

            # Leave-one-out: 每个样本的自身类原型排除自己
            raw_norm = F.normalize(raw_eeg, dim=1)
            centroids_norm = {k: F.normalize(c.unsqueeze(0), dim=1) for k, c in centroids.items()}
            proto_label_tensor = torch.tensor(proto_keys, device=self.device)

            preds = []
            for i in range(raw_eeg.shape[0]):
                sample = raw_norm[i]
                my_cls = labels[i].item()
                best_k, best_sim = -1, -1e9
                for k in proto_keys:
                    if k == my_cls and counts[k] > 1:
                        # 排除自身: proto = (sum - self) / (n-1)
                        proto = (centroids[k] * counts[k] - raw_eeg[i]) / (counts[k] - 1)
                        proto = F.normalize(proto.unsqueeze(0), dim=1)
                    else:
                        proto = centroids_norm[k]
                    sim = torch.matmul(sample.unsqueeze(0), proto.T).item()
                    if sim > best_sim:
                        best_sim, best_k = sim, k
                preds.append(best_k)
            pred_labels_proto = torch.tensor(preds, device=self.device)

            proto_acc = (pred_labels_proto == labels).float().mean()
            self.log('val/Proto_Acc', proto_acc, prog_bar=True)

            # Per-class Proto accuracy
            for lbl in unique_labels:
                mask = (labels == lbl)
                if mask.sum() > 0:
                    cls_acc = (pred_labels_proto[mask] == lbl).float().mean()
                    lbl_idx = lbl.item()
                    if 0 <= lbl_idx < len(EMOTION_NAMES):
                        name = EMOTION_NAMES[lbl_idx]
                        show_bar = name in ['Anger', 'Tenderness', 'Disgust', 'Fear']
                        self.log(f'val/Proto_{name}', cls_acc, prog_bar=show_bar)
        # =========================================================

        self.log_dict({
            'ext/val/loss': total_loss,
            f'ext/val/loss_{mode_name}': loss
        }, on_epoch=True, prog_bar=True)

        return total_loss

    def predict_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)):
            data = batch[0]
        else:
            data = batch
        fea = self(data.float())
        return fea

# =========================================================================
# 4. Lightning Module: MLPModel (Downstream Classification)
# =========================================================================

class MLPModel(pl.LightningModule):
    def __init__(self, model, cfg) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.lr = cfg.lr
        self.wd = cfg.wd
        self.criterion = torch.nn.CrossEntropyLoss()

        # --- Metrics ---
        num_classes = 9
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.val_kappa = CohenKappa(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)

    def training_step(self, batch, batch_idx):
        data, labels = batch
        logits = self.model(data)
        loss = self.criterion(logits, labels.long())

        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, labels)

        self.log('mlp/train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('mlp/train/acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        logits = self.model(data)
        loss = self.criterion(logits, labels.long())

        preds = torch.argmax(logits, dim=1)

        self.val_acc(preds, labels)
        self.val_f1(preds, labels)
        self.val_kappa(preds, labels)

        self.log_dict({
            'mlp/val/loss': loss,
            'mlp/val/acc': self.val_acc,
            'mlp/val/f1': self.val_f1,
            'mlp/val/kappa': self.val_kappa
        }, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        data, labels = batch
        logits = self(data)
        return logits.argmax(dim=1)