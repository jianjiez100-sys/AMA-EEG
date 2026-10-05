"""SEED pretraining from DAEST_SEED_Multiomdel_projector1.

Keep offline alignment, content masking, and class-averaged training alpha from
the SEED source. Validation alpha follows public AMA-EEG's per-sample CE rule.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .models import ResidualAdd
from .loss.seed_con_loss import MultiModalInfoNCELoss

def compute_entropy(logits):
    """
    Computes entropy of classification predictions.
    Entropy = - sum(p * log(p))
    """
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    entropy = -(probs * log_probs).sum(dim=1, keepdim=True)  # (Batch, 1)
    return entropy

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
        self.pretrain_mode = getattr(cfg, 'pretrain_mode', 0)

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
        # --- 残差投影头 (Residual Projector) ---
        self.text_projector = nn.Sequential(
            nn.Linear(1024, 1024),
            ResidualAdd(nn.Sequential(nn.GELU(), nn.Linear(1024, 1024), nn.Dropout(0.2))),
            nn.LayerNorm(1024),
        )
        self.image_projector = nn.Sequential(
            nn.Linear(1024, 1024),
            ResidualAdd(nn.Sequential(nn.GELU(), nn.Linear(1024, 1024), nn.Dropout(0.2))),
            nn.LayerNorm(1024),
        )

        # 根据配置设置 backbone 是否使用 LN-only 模式
        self.use_original_sampling = getattr(cfg, 'use_original_sampling', True)
        if hasattr(self.model, 'use_ln_backbone'):
            self.model.use_ln_backbone = not self.use_original_sampling
            print(f" [ExtractorModel] use_original_sampling={self.use_original_sampling}, "
                  f"use_ln_backbone={self.model.use_ln_backbone}")

        # Initialize components based on mode
        # if self.pretrain_mode == 2:
        #     # === Mode 2: Dynamic Fusion (Entropy + Probes) ===
        #     self.n_class = getattr(cfg, 'n_class', 9)
        #     feat_dim = 1024
        #     self.text_probe = nn.Linear(feat_dim, self.n_class)
        #  新增：在最外层获取类别数，优先从 cfg 取，没有就默认 3 (SEED)
        self.n_class = getattr(cfg, 'n_class', 3)

        # Initialize components based on mode
        if self.pretrain_mode == 2:
            # === Mode 2: Dynamic Fusion (Cross-Entropy + Probes) ===
            feat_dim = 1024
            self.text_probe = nn.Linear(feat_dim, self.n_class)
            self.image_probe = nn.Linear(feat_dim, self.n_class)
            self.probe_criterion = nn.CrossEntropyLoss()
            self.distill_criterion = MultiModalInfoNCELoss(
                temperature=cfg.loss_temp,
            )
            self.conf_temp = 0.05

        elif self.pretrain_mode == 3:
            # === Mode 3: Static Fusion (Fixed Alpha=0.5) ===
            self.distill_criterion = MultiModalInfoNCELoss(
                temperature=cfg.loss_temp,
            )

        # elif self.pretrain_mode == 0:
        #     # === Mode 0: Text  替换为 VideoSupConLoss ===
        #     self.criterion = MultiModalInfoNCELoss(temperature=0.1, threshold=0.85)
        elif self.pretrain_mode == 0:
            # === Mode 0: Text ===
            current_temp = getattr(cfg, 'loss_temp', 0.07)
            self.criterion = MultiModalInfoNCELoss(
                temperature=current_temp,
            )
            print(f"[Loss Init] Mode 0 Text: temp={current_temp}")

        else:
            # === Mode 1: Image ===
            current_temp = getattr(cfg, 'loss_temp', 0.07)
            self.criterion = MultiModalInfoNCELoss(
                temperature=current_temp,
            )
            print(f"[Loss Init] Mode 1 Image: temp={current_temp}")

    def forward(self, x):
        # Turn on saveFea during inference/finetuning
        if hasattr(self.model, 'set_saveFea'):
            self.model.set_saveFea(True)
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=self.max_epochs // self.restart_times, eta_min=0, last_epoch=-1
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, batch, batch_idx):
        # 1. 解包数据
        eeg, labels, txt_feat, img_feat, *extras = batch
        # event_ids: 唯一标识 (sub, vid, time)，用于 mask 同事件负样本
        event_ids = extras[0].view(-1) if extras else None
        # content_ids: 仅 (vid, time)，不含 sub，跨受试者共享内容标识
        content_ids = extras[1].view(-1) if len(extras) > 1 else None

        # 2. 类型转换与维度处理
        eeg = eeg.float()
        txt_feat = txt_feat.float()
        img_feat = img_feat.float()
        labels = labels.long()

        # 如果特征维度是 (Batch, Seq, Dim)，取平均变为 (Batch, Dim)
        if txt_feat.ndim == 3: txt_feat = txt_feat.mean(dim=1)
        if img_feat.ndim == 3: img_feat = img_feat.mean(dim=1)

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
            # === Mode 2: Dynamic Fusion (CE-based, averaged per emotion class) ===
            mode_name = "fusion_dynamic"

            # 1. 准备 Teacher 特征
            t_feat = F.normalize(txt_feat, dim=1)
            i_feat = F.normalize(img_feat, dim=1)

            # 2. 探针考核 (Probes)
            t_logits = self.text_probe(t_feat)
            i_logits = self.image_probe(i_feat)

            # 计算总的 Probe Loss 用于反向传播 (默认 reduction='mean')
            loss_probe_t = self.probe_criterion(t_logits, labels)
            loss_probe_i = self.probe_criterion(i_logits, labels)
            loss_probes = loss_probe_t + loss_probe_i

            # 3. 计算动态权重 (修改核心：基于 Loss 差异)
            with torch.no_grad():
                # A. 计算每个样本的独立 CrossEntropy Loss (不求平均)
                # 这样我们才能知道具体哪个样本、哪个模态预测得更准
                loss_func_none = nn.CrossEntropyLoss(reduction='none')
                loss_t_sample = loss_func_none(t_logits, labels)  # (Batch,)
                loss_i_sample = loss_func_none(i_logits, labels)  # (Batch,)

                # B. 计算 Loss 差异
                # 逻辑：如果 Image Loss (大) - Text Loss (小) > 0
                # 说明 Text 更准 -> diff > 0 -> Sigmoid > 0.5 -> Alpha 偏向 Text (符合预期)
                diff = (loss_i_sample - loss_t_sample) / self.conf_temp

                # 得到每个样本的瞬时 alpha (Batch, 1)
                inst_alpha = torch.sigmoid(diff).unsqueeze(1)

                # Preserve the SEED source: average within each emotion class.
                alpha_txt = torch.zeros_like(inst_alpha)
                unique_labels = torch.unique(labels)

                # FACED 9 类情感映射 (用于日志 Key)
                # emotion_names = [
                #     "0_Anger",  # neg_a
                #     "1_Disgust",  # neg_d
                #     "2_Fear",  # neg_f  <-- 对应 Alpha ≈ 0.22 (图像主导)
                #     "3_Sadness",  # neg_s
                #     "4_Neutral",  # neu
                #     "5_Amusement",  # pos_a
                #     "6_Inspiration",  # pos_i  <-- 对应 Alpha ≈ 0.94 (文本主导)
                #     "7_Joy",  # pos_j
                #     "8_Tenderness"  # pos_t
                # ]
                #  动态适配 SEED (3类) 和 FACED (9类)
                if self.n_class == 3:
                    emotion_names = [
                        "0_Negative",
                        "1_Neutral",
                        "2_Positive"
                    ]
                elif self.n_class == 9:
                    emotion_names = [
                        "0_Anger", "1_Disgust", "2_Fear", "3_Sadness", "4_Neutral",
                        "5_Amusement", "6_Inspiration", "7_Joy", "8_Tenderness"
                    ]
                else:
                    emotion_names = [f"Class_{i}" for i in range(self.n_class)]

                for label in unique_labels:
                    mask = (labels == label)

                    # 取该类别下所有样本 Alpha 的平均值
                    cls_alpha_mean = inst_alpha[mask.squeeze()].mean()

                    # 同一情感类别使用相同融合权重。
                    alpha_txt[mask.squeeze()] = cls_alpha_mean

                    # --- 记录日志 ---
                    label_idx = label.item()
                    if 0 <= label_idx < len(emotion_names):
                        cls_name = emotion_names[label_idx]
                    else:
                        cls_name = f"Class_{label_idx}"

                    log_data[f'alpha_class/{cls_name}'] = cls_alpha_mean

            # 4. 融合 (alpha_txt 已按当前 batch 中的情感类别分组平均)
            t_proj = F.normalize(self.text_projector(t_feat), dim=1)
            i_proj = F.normalize(self.image_projector(i_feat), dim=1)
            fused_target = alpha_txt * t_proj + (1.0 - alpha_txt) * i_proj
            fused_target = F.normalize(fused_target, dim=1)

            # 5. 蒸馏 (Distillation)
            proj_eeg = self.model(eeg, proj_mode='fusion')
            loss_distill = self.distill_criterion(proj_eeg, fused_target, content_ids=content_ids)
            loss = loss_distill + loss_probes

            # 记录平均 Alpha 和 分项 Loss
            log_data['ext/train/alpha_mean'] = alpha_txt.mean()
            log_data['ext/train/loss_distill'] = loss_distill
            log_data['ext/train/loss_probes'] = loss_probes

        elif self.pretrain_mode == 3:
            # === Mode 3: Static Fusion (0.3/0.7) ===
            # 你之前的代码这里改成了 0.3/0.7，我保留了这个修改
            mode_name = "fusion_static"

            t_feat = F.normalize(txt_feat.detach(), dim=1)
            i_feat = F.normalize(img_feat.detach(), dim=1)

            # 静态融合
            t_proj = F.normalize(self.text_projector(t_feat), dim=1)
            i_proj = F.normalize(self.image_projector(i_feat), dim=1)
            fused_target = 0.3 * t_proj + 0.7 * i_proj
            fused_target = F.normalize(fused_target, dim=1)

            proj_eeg = self.model(eeg, proj_mode='fusion')
            loss = self.distill_criterion(proj_eeg, fused_target, content_ids=content_ids)

        elif self.pretrain_mode == 0:
            # === Mode 0: Text Only（直接对齐原始文本特征，不做映射）===
            mode_name = "text"
            proj_txt = F.normalize(txt_feat, dim=1)
            proj_eeg = self.model(eeg, proj_mode='text')

            #  算 Loss 绝不能丢！千万别在这里写 self.criterion = ...
            loss = self.criterion(proj_eeg, proj_txt, content_ids=content_ids)

        elif self.pretrain_mode == 1:
            # === Mode 1: Image Only（直接对齐原始图像特征，不做映射）===
            mode_name = "image"
            proj_img = F.normalize(img_feat, dim=1)
            proj_eeg = self.model(eeg, proj_mode='image')
            loss = self.criterion(proj_eeg, proj_img, content_ids=content_ids)

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
        event_ids = extras[0].view(-1) if extras else None
        content_ids = extras[1].view(-1) if len(extras) > 1 else None

        eeg = eeg.float()
        txt_feat = txt_feat.float()
        img_feat = img_feat.float()
        # Public AMA-EEG uses validation labels for probe CE and dynamic alpha.
        # InfoNCE does not take emotion labels, but its dynamic target depends on them.
        labels = labels.long()

        if txt_feat.ndim == 3: txt_feat = txt_feat.mean(dim=1)
        txt_feat = F.normalize(txt_feat, dim=1)
        if img_feat.ndim == 3: img_feat = img_feat.mean(dim=1)
        img_feat = F.normalize(img_feat, dim=1)

        if hasattr(self.model, 'set_saveFea'):
            self.model.set_saveFea(False)

        loss = 0.0
        mode_name = ""
        target_feat = None  # 用于统一记录要对齐的目标特征

        # ================= 模式逻辑分支 =================
        if self.pretrain_mode == 2:
            mode_name = "fusion_dynamic"
            t_feat = F.normalize(txt_feat, dim=1)
            i_feat = F.normalize(img_feat, dim=1)

            t_logits = self.text_probe(t_feat)
            i_logits = self.image_probe(i_feat)

            # 计算验证集的探针 Loss (仅用于观察，不进总 Loss)
            loss_probe_t = self.probe_criterion(t_logits, labels)
            loss_probe_i = self.probe_criterion(i_logits, labels)
            self.log('val/probe_txt_loss', loss_probe_t, prog_bar=False)
            self.log('val/probe_img_loss', loss_probe_i, prog_bar=False)

            # Match public validation: per-sample CE alpha, without group averaging.
            with torch.no_grad():
                loss_func_none = nn.CrossEntropyLoss(reduction='none')
                loss_t_sample = loss_func_none(t_logits, labels)
                loss_i_sample = loss_func_none(i_logits, labels)
                diff = (loss_i_sample - loss_t_sample) / self.conf_temp
                alpha_txt = torch.sigmoid(diff).unsqueeze(1)

            t_proj = F.normalize(self.text_projector(t_feat), dim=1)
            i_proj = F.normalize(self.image_projector(i_feat), dim=1)
            fused_target = alpha_txt * t_proj + (1.0 - alpha_txt) * i_proj
            target_feat = F.normalize(fused_target, dim=1)

            proj_eeg = self.model(eeg, proj_mode='fusion')

            # Labels affect the target via alpha, not the content-ID negative mask.
            loss_distill = self.distill_criterion(proj_eeg, target_feat, content_ids=content_ids)
            loss = loss_distill

        elif self.pretrain_mode == 3:
            mode_name = "fusion_static"
            t_feat = F.normalize(txt_feat, dim=1)
            i_feat = F.normalize(img_feat, dim=1)

            t_proj = F.normalize(self.text_projector(t_feat), dim=1)
            i_proj = F.normalize(self.image_projector(i_feat), dim=1)
            fused_target = 0.5 * t_proj + 0.5 * i_proj
            target_feat = F.normalize(fused_target, dim=1)
            proj_eeg = self.model(eeg, proj_mode='fusion')

            # 坚决不传 labels
            loss = self.distill_criterion(proj_eeg, target_feat, content_ids=content_ids)

        elif self.pretrain_mode == 0:
            mode_name = "text"
            target_feat = F.normalize(txt_feat, dim=1)
            proj_eeg = self.model(eeg, proj_mode='text')

            loss = self.criterion(proj_eeg, target_feat, content_ids=content_ids)

        elif self.pretrain_mode == 1:
            mode_name = "image"
            target_feat = F.normalize(img_feat, dim=1)
            proj_eeg = self.model(eeg, proj_mode='image')
            loss = self.criterion(proj_eeg, target_feat, content_ids=content_ids)

        total_loss = self.w_clip * loss

        # =========================================================
        #  [新增监控指标：跨模态对齐效果体检]
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
            #  [新增]：零样本 9 分类情感准确率 (Zero-Shot Accuracy)
            # =====================================================
            # 逻辑：脑电找到最相似的目标特征 (文本/融合)，看看那个目标特征属于什么情感
            best_match_indices = sim_matrix.argmax(dim=1)
            pred_labels = labels[best_match_indices]
            zero_shot_acc = (pred_labels == labels).float().mean()

            # 记录到进度条和 TensorBoard，绝对不参与 loss 计算！
            self.log('val/ZeroShot_Acc', zero_shot_acc, prog_bar=True)
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
