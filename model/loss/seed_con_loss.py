"""SEED content-ID InfoNCE, preserved from the source experiment."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalInfoNCELoss(nn.Module):
    """
    专门用于多模态对齐的 Loss (EEG <-> Text/Image/Fusion)

     使用 content_id 掩码代替语义阈值掩码：
    相同 (视频, 时刻段) 的样本无论来自哪个受试者，都不作为负样本推开，
    因为他们共享同样的视频内容，语义上本应接近。
    """

    def __init__(self, temperature=0.15, symmetric=True):
        """
        Args:
            temperature: temperature coefficient (default 0.15).
            symmetric:   bidirectional (True) or unidirectional EEG->Target (False).
        """
        super(MultiModalInfoNCELoss, self).__init__()
        self.temperature = temperature
        self.symmetric = symmetric
        self.device = torch.device('cpu')
        self.criterion = nn.CrossEntropyLoss()

    def to(self, device):
        self.device = device
        self.criterion.to(device)
        return self

    def forward(self, preds, targets, content_ids=None):
        """
        Args:
            preds:       (Batch, Embed_Dim) -> EEG features
            targets:     (Batch, Embed_Dim) -> projected target features for logits
            content_ids: (Batch,) -> 同一 (视频, 时刻段) 的样本共享相同 content_id
                          不互相推开（即使来自不同受试者也不作为负样本）
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
        # logits 形状: (Batch_Size, Batch_Size)
        logits = torch.matmul(preds, targets.T) / self.temperature

        # 4. ================= [ content_id 掩码] =================
        # 相同 content_id  该对样本在 loss 中不互相推开
        with torch.no_grad():
            if content_ids is not None:
                content_ids = content_ids.view(-1)
                mask = content_ids.unsqueeze(0) == content_ids.unsqueeze(1)
                mask.fill_diagonal_(False)  # 对角线是正样本，不 mask
            else:
                mask = None

        if mask is not None:
            logits = logits.masked_fill(mask, -1e4)

        # 5. ================= [计算双向/单向 Loss] =================
        batch_size = preds.shape[0]
        labels = torch.arange(batch_size, dtype=torch.long).to(device)

        # 主方向：脑电找目标 (EEG -> Target)
        loss_e2t = self.criterion(logits, labels)

        # 6. ================= [单双向路由] =================
        if self.symmetric:
            # 开启双向：目标找脑电 (对称位置的掩码依然有效)
            loss_t2e = self.criterion(logits.T, labels)
            loss = (loss_e2t + loss_t2e) / 2
        else:
            loss = loss_e2t

        return loss
