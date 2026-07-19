import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from transformers.models.bert import BertModel, BertConfig


# =========================================================================
# 辅助模块
# =========================================================================
class ResidualAdd(nn.Module):
    """残差加法模块：out = x + residual(x)"""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


def stratified_layerNorm(out, n_samples):
    n_subs = int(out.shape[0] / n_samples)
    out_str = out.clone()
    for i in range(n_subs):
        out_oneSub = out[n_samples*i: n_samples*(i+1)]
        out_oneSub = out_oneSub.reshape(out_oneSub.shape[0], -1, out_oneSub.shape[-1]).permute(0,2,1)
        out_oneSub = out_oneSub.reshape(out_oneSub.shape[0]*out_oneSub.shape[1], -1)
        out_oneSub_str = (out_oneSub - out_oneSub.mean(dim=0)) / (out_oneSub.std(dim=0) + 1e-3)
        out_str[n_samples*i: n_samples*(i+1)] = out_oneSub_str.reshape(n_samples, -1, out_oneSub_str.shape[1]).permute(0,2,1).reshape(n_samples, out.shape[1], out.shape[2], -1)
    return out_str


def subject_aware_norm(inp, sub_ids, sub_means, sub_stds):
    """
    用离线预计算的受试者统计量做归一化，支持随机采样。
    仅用于 'initial' 位置（原始 EEG 输入）。

    Args:
        inp:       (B, 1, n_channs, timepoints)
        sub_ids:   (B,) long tensor，每个样本在 train_subs 列表中的位置索引
        sub_means: (n_train_subs, n_channs) 预计算的 channel-wise 均值
        sub_stds:  (n_train_subs, n_channs) 预计算的 channel-wise 标准差
    """
    means = sub_means[sub_ids]               # (B, n_channs)
    stds  = sub_stds[sub_ids]                # (B, n_channs)
    means = means.unsqueeze(1).unsqueeze(-1) # (B, 1, n_channs, 1)
    stds  = stds.unsqueeze(1).unsqueeze(-1)  # (B, 1, n_channs, 1)
    return (inp - means) / (stds + 1e-3)

# =========================================================================
# 核心模型：Conv_att_simple_new (修改版)
# =========================================================================
class Conv_att_simple_new(nn.Module):
    def __init__(self, n_timeFilters, timeFilterLen, n_msFilters, msFilter_timeLen, n_channs=64,
                 dilation_array=np.array([1, 3, 6, 12]), seg_att=30, avgPoolLen=30,
                 timeSmootherLen=6, multiFact=2, stratified=[], activ='softmax', temp=1.0, saveFea=True, has_att=True,
                 extract_mode='me', global_att=False,
                 clip_embed_dim=1024,  # 目标文本特征维度
                 image_embed_dim=1024,
                 text_embed_dim=1024,
                 proj_type='bottleneck',   # 'bottleneck' 或 'residual'
                 use_ln_backbone=False):   # True: 去掉BN+ReLU, 纯stratified_LN（原始设计）
        super().__init__()
        self.stratified = stratified
        self.msFilter_timeLen = msFilter_timeLen
        self.activ = activ
        self.temp = temp
        self.dilation_array = np.array(dilation_array)
        self.saveFea = saveFea
        self.has_att = has_att
        self.extract_mode = extract_mode
        self.global_att = global_att
        self.use_ln_backbone = use_ln_backbone
        self.return_layer = None  # None=backbone原始输出, 1-4=projector第n层输出
        self.return_pre_tconv = False  # True=跳过timeConv1/2, 直接输出attention后的256维特征

        # --- 1. Backbone: TSTC (时空卷积) ---
        self.timeConv = nn.Conv2d(1, n_timeFilters, (1, timeFilterLen), padding=(0, (timeFilterLen - 1) // 2))

        # 多尺度卷积 (Multi-scale Convolution)
        self.msConv1 = nn.Conv2d(n_timeFilters, n_timeFilters * n_msFilters, (n_channs, msFilter_timeLen),
                                 groups=n_timeFilters)
        self.msConv2 = nn.Conv2d(n_timeFilters, n_timeFilters * n_msFilters, (n_channs, msFilter_timeLen),
                                 dilation=(1, self.dilation_array[1]), groups=n_timeFilters)
        self.msConv3 = nn.Conv2d(n_timeFilters, n_timeFilters * n_msFilters, (n_channs, msFilter_timeLen),
                                 dilation=(1, self.dilation_array[2]), groups=n_timeFilters)
        self.msConv4 = nn.Conv2d(n_timeFilters, n_timeFilters * n_msFilters, (n_channs, msFilter_timeLen),
                                 dilation=(1, self.dilation_array[3]), groups=n_timeFilters)

        n_msFilters_total = n_timeFilters * n_msFilters * 4

        # 双分支 Attention: global_att='dual' 时局部窗口 + 全局通道并行融合
        self.dual_att = (isinstance(global_att, str) and global_att == 'dual')

        # --- 2. Backbone: Attention ---
        self.seg_att = seg_att
        self.att_conv = nn.Conv2d(n_msFilters_total, n_msFilters_total, (1, self.seg_att), groups=n_msFilters_total)
        self.att_pool = nn.AvgPool2d((1, self.seg_att), stride=1)
        self.att_pointConv = nn.Conv2d(n_msFilters_total, n_msFilters_total, (1, 1))

        # 全局 attention 分支：全时间维压缩，捕获秒级情绪偏侧化（仅 dual 模式启用）
        if self.dual_att:
            self.att_conv_global = nn.Conv2d(n_msFilters_total, n_msFilters_total,
                                             (1, self.seg_att), groups=n_msFilters_total)
            self.att_pointConv_global = nn.Conv2d(n_msFilters_total, n_msFilters_total, (1, 1))

        # --- 3. Feature Enhancer: TimeConv (现在是核心组件，不再是仅用于分类) ---
        self.avgpool = nn.AvgPool2d((1, avgPoolLen))

        # TimeConv1: 扩维 1 次 (x multiFact)
        self.timeConv1 = nn.Conv2d(n_msFilters_total, n_msFilters_total * multiFact, (1, timeSmootherLen),
                                   groups=n_msFilters_total)
        #加上batchnorm，使得数据服从于0-1之间的分布
        self.proj_bn1 = nn.BatchNorm2d(n_msFilters_total * multiFact)

        # TimeConv2: 扩维 2 次 (x multiFact)
        self.timeConv2 = nn.Conv2d(n_msFilters_total * multiFact, n_msFilters_total * multiFact * multiFact,
                                   (1, timeSmootherLen), groups=n_msFilters_total * multiFact)
        #加上batchnorm，使得数据服从于0-1之间的分布
        self.proj_bn2 = nn.BatchNorm2d(n_msFilters_total * multiFact * multiFact)

        # ================= [Pre-training Projectors] =================
        self.backbone_dropout = nn.Dropout2d(p=0.2)

        # 全局池化 (用于将 TimeConv2 输出的时序特征压缩为向量)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 🔥 [关键修正] 更新投影头的输入维度
        # 原始维度 * multiFact * multiFact (因为经过了两层扩维)
        self.backbone_out_dim = n_msFilters_total * multiFact * multiFact

        # 🔥 新增：定义预训练专用的非线性投影头
        # 维度变化：1024 -> 2048 -> 1024
        # 核心改造：引入非线性瓶颈层，防止 Projector 偷懒
        # self.projector = nn.Sequential(
        #     nn.Linear(1024, 256, bias=False),  # 压缩到 256
        #     nn.BatchNorm1d(256),  # BN 层极其重要，防止维度坍塌！
        #     nn.ReLU(inplace=True),  # 非线性激活
        #     nn.Linear(256, 1024, bias=True)  # 投影回文本维度
        # )
        # --- EEG 端投影头 ---
        def _make_projector():
            if proj_type == 'residual':
                return nn.Sequential(
                    nn.Linear(1024, 1024),
                    ResidualAdd(nn.Sequential(
                        nn.GELU(),
                        nn.Linear(1024, 1024),
                        nn.Dropout(0.2),
                    )),
                    nn.LayerNorm(1024),
                )
            else:  # 'bottleneck' (默认)
                return nn.Sequential(
                    nn.Linear(1024, 2048, bias=False),
                    nn.BatchNorm1d(2048),
                    nn.GELU(),
                    nn.Dropout(0.2),
                    nn.Linear(2048, 2048, bias=False),
                    nn.BatchNorm1d(2048),
                    nn.GELU(),
                    nn.Dropout(0.2),
                    nn.Linear(2048, 1024, bias=True),
                )

        self.projector = _make_projector()

        print(
            f"⚡ [Model Init] Projector Input Dim calculated as: {self.backbone_out_dim} (Original: {n_msFilters_total} x {multiFact}^2)")

    def forward(self, input, proj_mode='fusion'):
        # 0. 初始归一化 (可选)
        if 'initial' in self.stratified:
            input = stratified_layerNorm(input, int(input.shape[0] / 2))

        # 1. TSTC 主干提取
        out = self.timeConv(input)
        p = self.dilation_array * (self.msFilter_timeLen - 1)
        out1 = self.msConv1(F.pad(out, (int(p[0] // 2), p[0] - int(p[0] // 2)), "constant", 0))
        out2 = self.msConv2(F.pad(out, (int(p[1] // 2), p[1] - int(p[1] // 2)), "constant", 0))
        out3 = self.msConv3(F.pad(out, (int(p[2] // 2), p[2] - int(p[2] // 2)), "constant", 0))
        out4 = self.msConv4(F.pad(out, (int(p[3] // 2), p[3] - int(p[3] // 2)), "constant", 0))
        out = torch.cat((out1, out2, out3, out4), 1)

        # === 注意力机制 ===
        if self.has_att:
            if self.dual_att:
                # 局部分支：seg_att 窗口内的时序注意力，捕获 ~120ms 快速振荡模式
                att_local = F.relu(self.att_conv(F.pad(out, (self.seg_att - 1, 0), "constant", 0)))
                att_local = self.att_pool(F.pad(att_local, (self.seg_att - 1, 0), "constant", 0))
                att_local = self.att_pointConv(att_local)

                # 全局分支：全时间维压缩为通道注意力，捕获秒级 α 不对称性等慢变模式
                att_global = F.relu(self.att_conv_global(
                    F.pad(out, (self.seg_att - 1, 0), "constant", 0)))
                att_global = torch.mean(att_global, dim=-1, keepdim=True)  # (B,256,T,1)
                att_global = self.att_pointConv_global(att_global)

                # 固定加法融合（策略1），无额外融合参数
                # pointConv 权重会在训练中隐式学习两分支的相对重要性
                att_w = att_local + att_global
            else:
                # 原始单分支逻辑（global_att=False/True 均兼容）
                att_w = F.relu(self.att_conv(F.pad(out, (self.seg_att - 1, 0), "constant", 0)))
                if self.global_att:
                    att_w = torch.mean(F.pad(att_w, (self.seg_att - 1, 0), "constant", 0), -1).unsqueeze(-1)
                else:
                    att_w = self.att_pool(F.pad(att_w, (self.seg_att - 1, 0), "constant", 0))
                att_w = self.att_pointConv(att_w)

            if self.activ == 'relu':
                att_w = F.relu(att_w)
            elif self.activ == 'softmax':
                att_w = F.softmax(att_w / self.temp, dim=1)
            elif self.activ == 'sigmoid':
                att_w = torch.sigmoid(att_w)
            out = att_w * F.relu(out)
        else:
            if self.extract_mode == 'me': out = F.relu(out)

        out = self.backbone_dropout(out)

        # =========================================================
        # 🔥 [关键逻辑] 统一处理流：预训练和分类都走这里
        # =========================================================

        # A. 池化
        if self.extract_mode == 'de':
            out = F.relu(out)
        out = self.avgpool(out)

        # 消融实验：跳过 timeConv1/2，直接输出 attention 后的 256 维特征
        if self.saveFea and self.return_pre_tconv:
            out = self.global_pool(out)
            return out.flatten(start_dim=1)

        # B. TimeConv1 (含分层归一化可选)
        if 'middle1' in self.stratified:
            out = stratified_layerNorm(out, int(out.shape[0] / 2))
        out = self.timeConv1(out)
        if self.use_ln_backbone:
            # 原始设计: 纯 stratified_LN，不加 BN+ReLU
            pass
        else:
            out = self.proj_bn1(out)  # 后加的 BN
            out = F.relu(out)         # 后加的 ReLU

        out = self.timeConv2(out)
        if self.use_ln_backbone:
            pass  # 原始设计: 不加 BN+ReLU
        else:
            out = self.proj_bn2(out)
            out = F.relu(out)
        if 'middle2' in self.stratified:
            out = stratified_layerNorm(out, int(out.shape[0] / 2))


        out = self.global_pool(out)
        out_flat = out.flatten(start_dim=1)

        if self.saveFea:
            # 下游特征提取模式
            # return_layer=None/0: 返回 backbone 原始 1024 维
            # return_layer=1-4:   返回 projector 第 n 层输出（均为 1024 维）
            if self.return_layer is not None and self.return_layer > 0:
                x = out_flat
                # projector: 4 层, 前三层各 4 子模块(Linear/BN/GELU/Dropout), 末层 1 子模块(Linear)
                n_sub = min(self.return_layer * 4, 13)
                for i in range(n_sub):
                    x = self.projector[i](x)
                return x
            return out_flat
        else:
            # === 预训练 (Contrastive Learning) ===
            projected_fea = self.projector(out_flat)
            return F.normalize(projected_fea, dim=1)

    # Setters 用于外部控制
    def set_saveFea(self, saveFea):
        self.saveFea = saveFea

    def set_extract_mode(self, extract_mode):
        self.extract_mode = extract_mode

    def set_stratified(self, stratified):
        self.stratified = stratified

    def set_return_layer(self, layer):
        self.return_layer = layer

    def set_return_pre_tconv(self, flag):
        self.return_pre_tconv = flag

class simpleNN3(nn.Module):
    def __init__(self, inp_dim, hidden_dim=[128,64], out_dim=9, dropout=0.2, bn='no'):
        super(simpleNN3, self).__init__()
        self.bn = bn
        self.drop = nn.Dropout(p=dropout)

        # 动态构建隐藏层: 支持任意长度 hidden_dim
        # hidden_dim=[]      → 纯线性分类 (inp_dim → out_dim)
        # hidden_dim=[32]    → 2层 MLP (inp_dim → 32 → out_dim)
        # hidden_dim=[128,64] → 3层 MLP (inp_dim → 128 → 64 → out_dim, 原默认)
        self.hidden_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()  # BN 或 LN 统一存储
        prev_dim = inp_dim
        for h_dim in hidden_dim:
            self.hidden_layers.append(nn.Linear(prev_dim, h_dim))
            if bn.startswith('ln'):
                self.norm_layers.append(nn.LayerNorm(h_dim, elementwise_affine=False))
            else:
                self.norm_layers.append(nn.BatchNorm1d(h_dim, affine=False))
            prev_dim = h_dim

        self.num_hidden = len(hidden_dim)
        self.output_layer = nn.Linear(prev_dim, out_dim)

    def forward(self, input):
        out = input
        for i in range(self.num_hidden):
            out = self.hidden_layers[i](out)
            # 归一化规则:
            #   'no'    = 不使用归一化
            #   'bn1'   = 仅第 0 层用 BN
            #   'bn2'   = 第 0,1 层用 BN
            #   'all'   = 所有层用 BN
            #   'ln1'   = 仅第 0 层用 LN
            #   'ln2'   = 第 0,1 层用 LN
            #   'ln_all' = 所有层用 LN
            use_norm = False
            if self.bn == 'all' or self.bn == 'ln_all':
                use_norm = True
            elif (self.bn in ['bn1', 'ln1']) and i == 0:
                use_norm = True
            elif (self.bn in ['bn2', 'ln2']) and i <= 1:
                use_norm = True
            if use_norm:
                out = self.norm_layers[i](out)
            out = F.relu(out)
            out = self.drop(out)
        out = self.output_layer(out)
        return out
