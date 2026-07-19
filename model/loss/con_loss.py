import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCLRLoss(nn.Module):
    def __init__(self, temperature):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature
        self.CEL = torch.nn.CrossEntropyLoss()
        self.device = torch.device('cpu')

    def to(self, device):
        self.device = device
        self.CEL = self.CEL.to(device)
        return self

    def info_nce_loss(self, features):
        device = self.device
        bs = int(features.shape[0] // 2)
        labels = torch.cat([torch.arange(bs) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(device)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([negatives, positives], dim=1)
        labels = torch.ones(logits.shape[0], dtype=torch.long) * (logits.shape[1] - 1)
        labels = labels.to(device)

        logits = logits / self.temperature
        return logits, labels

    def forward(self, features):
        self.to(features.device)
        logits, labels = self.info_nce_loss(features)
        loss = self.CEL(logits, labels)
        return loss, logits, labels


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf."""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.device = torch.device('cpu')

    def to(self, device):
        self.device = device
        return self

    def forward(self, features, labels=None, mask=None, modified=False):
        #device = self.device
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask

        if modified:
            nega_exp_logits_sum = (exp_logits * (~mask.bool())).sum(1)
            log_prob = torch.zeros_like(logits)
            for i in range(logits.shape[0]):
                for j in torch.nonzero(mask[i]).squeeze():
                    log_prob[i, j] = logits[i, j] - torch.log(nega_exp_logits_sum[i] + exp_logits[i, j])
        else:
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss, exp_logits, mask


# class MultiModalInfoNCELoss(nn.Module):
#     """
#     专门用于多模态对齐的 Loss (EEG <-> Text 或 EEG <-> Image 或 EEG <-> FusedTarget)
#     参考 CLIP 的损失函数设计: Symmetric Cross Entropy
#     """
#
#     def __init__(self, temperature=0.07):
#         super(MultiModalInfoNCELoss, self).__init__()
#         self.temperature = temperature
#         self.device = torch.device('cpu')
#         self.criterion = nn.CrossEntropyLoss()
#
#     def to(self, device):
#         self.device = device
#         self.criterion.to(device)
#         return self
#
#     def forward(self, preds, targets):
#         """
#         Args:
#             preds:   (Batch, Embed_Dim) -> EEG Features
#             targets: (Batch, Embed_Dim) -> Text/Image/Fused Features
#         """
#         device = preds.device
#
#         # ================= [关键修复] 强制展平为 2D =================
#         # 能够处理 (B, 1, D) 或 (B, D, 1, 1) 等各种奇葩维度
#         if preds.dim() > 2:
#             preds = preds.view(preds.size(0), -1)
#         if targets.dim() > 2:
#             targets = targets.view(targets.size(0), -1)
#         # ========================================================
#
#         # 2. 归一化 (Cosine Similarity 前置步骤)
#         preds = F.normalize(preds, dim=1)
#         targets = F.normalize(targets, dim=1)
#
#         # 3. 计算相似度矩阵 (Batch x Batch)
#         logits = torch.matmul(preds, targets.T) / self.temperature
#
#         # 4. 生成对角线标签 (Batch, )
#         batch_size = preds.shape[0]
#         labels = torch.arange(batch_size, dtype=torch.long).to(device)
#
#         # 5. 计算双向 Loss (Symmetric)
#         loss_e2t = self.criterion(logits, labels)
#         loss_t2e = self.criterion(logits.T, labels)
#
#         # 6. 平均
#         loss = (loss_e2t + loss_t2e) / 2
#
#         return loss


class MultiModalInfoNCELoss(nn.Module):
    """
    专门用于多模态对齐的 Loss (EEG <-> Text)
    🔥 升级版：引入 Text-Guided Similarity Masking，防止假阴性（False Negatives）干扰
    """

    def __init__(self, temperature=0.15, threshold=1.0, symmetric=True,
                 mask_mode='vid_sub'):
        """
        Args:
            temperature: 温度系数。建议设为 0.1，给类内特征留出聚类空间。
            threshold:   相似度阈值。仅 mask_mode='threshold' 时生效。
            symmetric:   是否开启双向对比 (True: 双向 EEG<->Text, False: 单向 EEG->Text)。
            mask_mode:   假阴性掩码模式:
                         'vid_sub'   — 基于视频+受试者ID (同视频+不同受试者→mask)
                         'threshold' — 基于文本相似度阈值 (text_sim > threshold → mask)
                         'none'      — 不掩码
        """
        # if pretrain_mode == 0:
        #     # 📝 文本模式：使用低阈值，保护队友！
        #     # 结合你之前的统计，同类文本均值约 0.5，所以设为 0.45 甚至 0.4 最好
        #     threshold = MultiModalInfoNCELoss(temperature=0.1, threshold=0.48)
        #     针对real_describe使用的是0.48，对于这里最新的clear_divide使用的是0.40
        #
        # elif pretrain_mode == 1:
        #     # 🖼️ 图像模式：使用高阈值，过滤相邻帧冗余！
        #     threshold = MultiModalInfoNCELoss(temperature=0.1, threshold=0.78)
        # 可以将这里的图像的掩码参数设置为0.78，主要目的是不要让同一视频下面的样本相互排斥
        super(MultiModalInfoNCELoss, self).__init__()
        self.temperature = temperature
        self.threshold = threshold
        self.symmetric = symmetric
        self.mask_mode = mask_mode
        self.device = torch.device('cpu')
        self.criterion = nn.CrossEntropyLoss()

    def to(self, device):
        self.device = device
        self.criterion.to(device)
        return self

    def forward(self, preds, targets, original_targets=None, vid_ids=None, sub_ids=None):
        """
        Args:
            preds:            (Batch, Embed_Dim) -> 脑电特征 (EEG Features)
            targets:          (Batch, Embed_Dim) -> 投影后的目标特征，用于计算 logits
            original_targets: (Batch, Embed_Dim) -> 原始目标特征 (已弃用, 保留兼容)
            vid_ids:          (Batch,) -> 视频 ID. 同视频→同文本→假阴性, 需 mask.
            sub_ids:          (Batch,) -> 受试者 ID. 仅 mask 同视频+不同受试者.
                             同视频+同受试者+不同时刻 → 仍为有效负样本.
        """
        device = preds.device

        # 1. ================= [强制展平为 2D] =================
        if preds.dim() > 2:
            preds = preds.view(preds.size(0), -1)
        if targets.dim() > 2:
            targets = targets.view(targets.size(0), -1)

        # 2. ================= [归一化] =================
        preds = F.normalize(preds, dim=1)
        targets = F.normalize(targets, dim=1)

        # 3. ================= [计算相似度矩阵] =================
        logits = torch.matmul(preds, targets.T) / self.temperature

        # 4. ================= [🔥 掩码 (由 mask_mode 控制)] =================
        with torch.no_grad():
            if self.mask_mode == 'vid_sub':
                if vid_ids is not None and sub_ids is not None:
                    # 只 mask 同视频+不同受试者 (假阴性, 同一文本被多个受试者共享)
                    vid_ids = vid_ids.view(-1)
                    sub_ids = sub_ids.view(-1)
                    same_vid = torch.eq(vid_ids.unsqueeze(0), vid_ids.unsqueeze(1))
                    same_sub = torch.eq(sub_ids.unsqueeze(0), sub_ids.unsqueeze(1))
                    mask = same_vid & (~same_sub)
                    mask.fill_diagonal_(False)
                elif vid_ids is not None:
                    # 降级: 仅 vid_ids → mask 所有同视频对
                    vid_ids = vid_ids.view(-1)
                    mask = torch.eq(vid_ids.unsqueeze(0), vid_ids.unsqueeze(1))
                    mask.fill_diagonal_(False)
                else:
                    mask = torch.zeros_like(logits, dtype=torch.bool)

            elif self.mask_mode == 'threshold':
                if original_targets is not None:
                    if original_targets.dim() > 2:
                        original_targets = original_targets.view(original_targets.size(0), -1)
                    mask_source = F.normalize(original_targets, dim=1)
                else:
                    mask_source = targets
                text_sim = torch.matmul(mask_source, mask_source.T)
                mask = text_sim > self.threshold
                mask.fill_diagonal_(False)

            else:  # 'none' 或未知
                mask = torch.zeros_like(logits, dtype=torch.bool)

        # 将掩码位置的 Logits 替换为一个极小的负数
        # 这样在算 Softmax 时，这些位置的权重会趋近于 0
        # 使用 -1e4 是为了兼容 bfloat16 精度，防止溢出
        logits = logits.masked_fill(mask, -1e4)

        # 5. ================= [计算双向/单向 Loss] =================
        batch_size = preds.shape[0]
        labels = torch.arange(batch_size, dtype=torch.long).to(device)

        # 主方向：脑电找文本 (EEG -> Text) 始终计算
        loss_e2t = self.criterion(logits, labels)

        # 6. ================= [单双向路由] =================
        if self.symmetric:
            # 开启双向：文本找脑电 (对称位置的掩码依然有效)
            loss_t2e = self.criterion(logits.T, labels)
            # 平均
            loss = (loss_e2t + loss_t2e) / 2
        else:
            # 开启单向：仅利用 EEG 追逐文本质心的拉力，避免文本对脑电的内部排斥
            loss = loss_e2t

        return loss

class CrossModalSupConLoss(nn.Module):
    """
    跨模态专用 SupCon Loss：只在 EEG×Text 矩阵上计算，
    完全不包含 EEG↔EEG 或 Text↔Text 的 intra-modal 对。
    同情感的 (EEG_i, Text_j) 视为正样本，从根本上消除假阴性。

    Loss = 0.5 * L(EEG→Text) + 0.5 * L(Text→EEG)
    其中 L(EEG→Text): 每条 EEG_i 在所有 Text 中找同情感的 Text_j
    """

    def __init__(self, temperature=0.07):
        super(CrossModalSupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, preds, targets, labels):
        """
        Args:
            preds:   (B, D) → EEG 特征
            targets: (B, D) → Text 特征
            labels:  (B,)   → 情感类别标签
        """
        device = preds.device
        B = preds.shape[0]

        if preds.dim() > 2:
            preds = preds.view(B, -1)
        if targets.dim() > 2:
            targets = targets.view(targets.size(0), -1)

        preds   = F.normalize(preds,   dim=1)   # (B, D)
        targets = F.normalize(targets, dim=1)   # (B, D)

        # 跨模态相似度矩阵 (B, B)：sim[i,j] = cosine(EEG_i, Text_j)
        sim = torch.matmul(preds, targets.T) / self.temperature

        # 正样本掩码：同情感 = 正样本
        # pos_mask[i,j] = 1 当且仅当 labels[i] == labels[j]
        labels_col = labels.view(-1, 1)
        pos_mask = torch.eq(labels_col, labels_col.T).float().to(device)  # (B, B)

        # --- EEG → Text 方向 ---
        sim_e2t = sim - sim.max(dim=1, keepdim=True).values.detach()  # 数值稳定
        exp_e2t = torch.exp(sim_e2t)
        log_prob_e2t = sim_e2t - torch.log(exp_e2t.sum(dim=1, keepdim=True))
        n_pos_e = pos_mask.sum(dim=1).clamp(min=1)
        loss_e2t = -(pos_mask * log_prob_e2t).sum(dim=1) / n_pos_e
        loss_e2t = loss_e2t.mean()

        # --- Text → EEG 方向（cosine 对称，直接转置）---
        # sim_t2e[a,b] = cosine(Text_a, EEG_b)；row=Text anchor，col=EEG target
        sim_t2e = sim.T - sim.T.max(dim=1, keepdim=True).values.detach()
        exp_t2e = torch.exp(sim_t2e)
        log_prob_t2e = sim_t2e - torch.log(exp_t2e.sum(dim=1, keepdim=True))
        # pos_mask[a,b] = (labels[a]==labels[b])，语义上 Text_a 的正样本 EEG_b，dim=1 沿列求和
        n_pos_t = pos_mask.sum(dim=1).clamp(min=1)
        loss_t2e = -(pos_mask * log_prob_t2e).sum(dim=1) / n_pos_t
        loss_t2e = loss_t2e.mean()

        return (loss_e2t + loss_t2e) / 2


class MultiModalSupConLoss(nn.Module):
    """
    专门用于多模态上限测试的 SupConLoss 包装器。
    将 EEG 和 Text 视为同一个样本的 2 个视图 (n_views=2)。
    """
    def __init__(self, temperature=0.10):
        super(MultiModalSupConLoss, self).__init__()
        # 初始化基础的 SupConLoss
        self.supcon = SupConLoss(temperature=temperature)

    def to(self, device):
        self.device = device
        self.supcon.to(device)
        return self

    def forward(self, preds, targets, labels):
        """
        Args:
            preds:   (Batch, Embed_Dim) -> 脑电特征 (EEG Features)
            targets: (Batch, Embed_Dim) -> 文本特征 (Text Features)
            labels:  (Batch,) -> 真实的 9 分类情感标签 (Ground Truth Labels)
        """
        # 1. ================= [强制展平为 2D] =================
        if preds.dim() > 2:
            preds = preds.view(preds.size(0), -1)
        if targets.dim() > 2:
            targets = targets.view(targets.size(0), -1)

        # 2. ================= [归一化] =================
        preds = F.normalize(preds, dim=1)
        targets = F.normalize(targets, dim=1)

        # 3. ================= [构建多视图 Features] =================
        # SupConLoss 要求的维度是 [Batch_Size, n_views, Embed_Dim]
        # 我们把 EEG 作为视图 1，Text 作为视图 2 进行堆叠
        # torch.stack 会在 dim=1 处新增一个维度，形状变为 (Batch_Size, 2, Embed_Dim)
        multimodal_features = torch.stack([preds, targets], dim=1)

        # 4. ================= [计算 SupCon Loss] =================
        # 调用基础的 SupConLoss，传入组合好的特征和真实情感标签
        loss, _, _ = self.supcon(features=multimodal_features, labels=labels)

        return loss


class VideoSupConLoss(nn.Module):
    """
    以视频 ID 为正样本定义的跨模态 SupCon Loss。

    正样本：同一视频下所有 (EEG_i, Target_j) 对（不推远）
    负样本：不同视频的所有样本（正常推远）

    实现方式：
    - 构造 vid_id 相同 → 正样本 mask
    - 正样本从分母中移除（SupCon 标准做法），只有负样本产生斥力
    - 双向对称：EEG→Target + Target→EEG
    """

    def __init__(self, temperature=0.07):
        super(VideoSupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, preds, targets, vid_ids):
        """
        Args:
            preds:    (B, D) → EEG 投影特征
            targets:  (B, D) → Text/Image 投影特征
            vid_ids:  (B,)   → 视频 ID，同 vid_id 视为正样本
        """
        device = preds.device
        B = preds.shape[0]

        if preds.dim() > 2:
            preds = preds.view(B, -1)
        if targets.dim() > 2:
            targets = targets.view(targets.size(0), -1)

        preds   = F.normalize(preds,   dim=1)
        targets = F.normalize(targets, dim=1)

        # 跨模态相似度矩阵 (B, B): sim[i,j] = cosine(EEG_i, Target_j)
        sim = torch.matmul(preds, targets.T) / self.temperature

        # 正样本 mask：同 vid_id → True，对角线也包含在内
        vid_ids = vid_ids.view(-1)
        pos_mask = torch.eq(vid_ids.unsqueeze(0), vid_ids.unsqueeze(1)).float().to(device)  # (B, B)

        # 负样本 mask：不同 vid_id，用于构造分母
        neg_mask = 1.0 - pos_mask  # (B, B)

        # --- EEG → Target 方向 ---
        sim_e2t = sim - sim.max(dim=1, keepdim=True).values.detach()
        exp_e2t = torch.exp(sim_e2t)

        # 分母只包含负样本（同视频样本不推远，即不进分母）
        denom_e2t = (exp_e2t * neg_mask).sum(dim=1, keepdim=True).clamp(min=1e-8)
        log_prob_e2t = sim_e2t - torch.log(denom_e2t)

        # 分子：对所有正样本求平均（包括对角线自身）
        n_pos = pos_mask.sum(dim=1).clamp(min=1)
        loss_e2t = -(pos_mask * log_prob_e2t).sum(dim=1) / n_pos
        loss_e2t = loss_e2t.mean()

        # --- Target → EEG 方向（对称）---
        sim_t2e = sim.T - sim.T.max(dim=1, keepdim=True).values.detach()
        exp_t2e = torch.exp(sim_t2e)

        denom_t2e = (exp_t2e * neg_mask).sum(dim=1, keepdim=True).clamp(min=1e-8)
        log_prob_t2e = sim_t2e - torch.log(denom_t2e)

        n_pos_t = pos_mask.sum(dim=1).clamp(min=1)
        loss_t2e = -(pos_mask * log_prob_t2e).sum(dim=1) / n_pos_t
        loss_t2e = loss_t2e.mean()

        return (loss_e2t + loss_t2e) / 2

class SemanticVideoSupConLoss(nn.Module):
    """
    语义感知的动态视频级 SupCon Loss (解决特征过平滑问题)

    判断逻辑：
    - 正样本 (Pull): 同一视频 + 文本语义相似度 > threshold (情绪稳定期，需平滑去噪)
    - 负样本 (Push): 不同视频 OR (同一视频 + 文本语义相似度 <= threshold) (剧情突变，需锐化边界)
    """

    def __init__(self, temperature=0.1, text_threshold=0.45):
        """
        Args:
            temperature: 对比学习温度系数 (参考你之前 InfoNCE 用的 0.1 或 0.15)
            text_threshold: 语义断崖阈值。基于统计，FACED 推荐 0.45。
        """
        super(SemanticVideoSupConLoss, self).__init__()
        self.temperature = temperature
        self.text_threshold = text_threshold

    def forward(self, preds, targets, vid_ids, original_txt_feats):
        """
        Args:
            preds: (B, D) -> EEG 投影特征
            targets: (B, D) -> Text 投影特征
            vid_ids: (B,) -> 视频 ID 标签
            original_txt_feats: (B, D_txt) -> 原始未投影的 CLIP 文本特征 (作为语义裁判)
        """
        device = preds.device
        B = preds.shape[0]

        # 1. ==== 维度安全检查 ====
        if preds.dim() > 2: preds = preds.view(B, -1)
        if targets.dim() > 2: targets = targets.view(B, -1)
        if original_txt_feats.dim() > 2: original_txt_feats = original_txt_feats.view(B, -1)

        # 2. ==== 特征归一化 ====
        preds = F.normalize(preds, dim=1)
        targets = F.normalize(targets, dim=1)
        orig_txt = F.normalize(original_txt_feats.detach(), dim=1)

        # 3. ==== 裁判入场：计算原始文本的全局相似度矩阵 ====
        # text_sim_matrix: (B, B)
        with torch.no_grad():
            text_sim_matrix = torch.matmul(orig_txt, orig_txt.T)

        # 4. ==== 跨模态特征相似度矩阵 ====
        # sim[i,j] = cosine(EEG_i, Target_j)
        sim = torch.matmul(preds, targets.T) / self.temperature

        # 5. ==== 构建动态的正负样本 Mask ====
        vid_ids = vid_ids.view(-1)
        # 条件A：是否来自同一个视频
        same_video_mask = torch.eq(vid_ids.unsqueeze(0), vid_ids.unsqueeze(1))
        # 条件B：语义是否维持在阈值之上 (情绪连贯)
        semantic_sim_mask = text_sim_matrix > self.text_threshold

        # 综合判定：同视频 且 语义连贯 的才是正样本
        pos_mask = (same_video_mask & semantic_sim_mask).float().to(device)

        # 🚨 保命设定：对角线（自己和自己）必须无条件是正样本
        pos_mask.fill_diagonal_(1.0)

        # 负样本：其余全为负样本（自然包含了同视频但语义突变的样本，强迫模型推开它们！）
        neg_mask = 1.0 - pos_mask

        # ==============================================================
        # 6. ==== 计算 SupCon 损失 (EEG -> Text) ====
        # ==============================================================
        # 减去最大值防止 exp 溢出
        sim_e2t = sim - sim.max(dim=1, keepdim=True).values.detach()
        exp_e2t = torch.exp(sim_e2t)

        # 分母：仅计算 neg_mask 标记的敌人 (同视频突变帧 + 异视频帧)
        denom_e2t = (exp_e2t * neg_mask).sum(dim=1, keepdim=True).clamp(min=1e-8)
        log_prob_e2t = sim_e2t - torch.log(denom_e2t)

        # 分子：将所有 pos_mask 标记的战友拉近
        n_pos = pos_mask.sum(dim=1).clamp(min=1)
        loss_e2t = -(pos_mask * log_prob_e2t).sum(dim=1) / n_pos
        loss_e2t = loss_e2t.mean()

        # ==============================================================
        # 7. ==== 计算 SupCon 损失 (Text -> EEG) 对称拉扯 ====
        # ==============================================================
        sim_t2e = sim.T - sim.T.max(dim=1, keepdim=True).values.detach()
        exp_t2e = torch.exp(sim_t2e)

        denom_t2e = (exp_t2e * neg_mask).sum(dim=1, keepdim=True).clamp(min=1e-8)
        log_prob_t2e = sim_t2e - torch.log(denom_t2e)

        n_pos_t = pos_mask.sum(dim=1).clamp(min=1)
        loss_t2e = -(pos_mask * log_prob_t2e).sum(dim=1) / n_pos_t
        loss_t2e = loss_t2e.mean()

        # 返回双向平均 Loss
        return (loss_e2t + loss_t2e) / 2