import os
import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import pytorch_lightning as pl
# 👇 必须导入 Callback 基类
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
import logging
import sys
# Used to get original path, escaping Hydra's sandbox
from hydra.utils import get_original_cwd

# Import DataModule and LightningModule
from data.pl_datamodule import FACEDDataModule, EEGDataModule
from model.pl_models import ExtractorModel

# Set matrix multiplication precision
torch.set_float32_matmul_precision('medium')

log = logging.getLogger(__name__)


# =========================================================================
# 👇 新增：自定义回调函数，用于打印 9 类情感权重
# =========================================================================
class EmotionWeightLogger(Callback):
    """
    专门用于在每个 Epoch 结束时，打印 9 类情感权重的详细表格
    """

    def on_train_epoch_end(self, trainer, pl_module):
        # 1. 获取当前 Epoch 的所有指标
        metrics = trainer.callback_metrics

        # 2. 筛选出以 'alpha_class/' 开头的指标
        alpha_data = {k: v for k, v in metrics.items() if k.startswith("alpha_class/")}

        if not alpha_data:
            return

        # 3. 打印表头
        print(f"\n{'=' * 60}")
        print(f"📊 Fold {trainer.current_epoch} - 情感类别权重报告 (Class-Level Alpha)")
        print(f"{'-' * 60}")
        print(f"{'Emotion Class':<20} | {'Text Weight (Alpha)':<20} | {'Image Weight':<20}")
        print(f"{'-' * 60}")

        # 4. 排序并打印每一行
        # key 格式是 alpha_class/0_Amusement -> 提取出 0_Amusement
        sorted_keys = sorted(alpha_data.keys())

        for key in sorted_keys:
            emotion_name = key.replace("alpha_class/", "")
            alpha_val = alpha_data[key].item()  # 转为 python float

            # alpha 是文本权重，1-alpha 是图像权重
            print(f"{emotion_name:<20} | {alpha_val:<20.4f} | {1 - alpha_val:<20.4f}")

        print(f"{'=' * 60}\n")


class ValMetricsLogger(Callback):
    """在每个 epoch 结束时打印完整的验证指标，避免进度条截断"""

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        val_keys = sorted([k for k in metrics.keys() if k.startswith('val/') or k.startswith('ext/val/')])

        if not val_keys:
            return

        epoch = trainer.current_epoch
        print(f"\n{'─' * 55}")
        print(f"📋 Epoch {epoch:>3} 验证指标:")
        # 每行放 3 个指标
        row = []
        for key in val_keys:
            try:
                val = metrics[key].item()
                row.append(f"{key:<28} {val:.4f}")
            except:
                pass
            if len(row) == 2:
                print(f"   {'  |  '.join(row)}")
                row = []
        if row:
            print(f"   {'  |  '.join(row)}")
        print(f"{'─' * 55}")


@hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
def train_ext(cfg: DictConfig) -> None:
    # 1. 设置随机种子
    pl.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ================= [配置: 获取预训练模式] =================
    # 0: Text, 1: Image, 2: Auto-Fusion
    pretrain_mode = cfg.train.get('pretrain_mode', 0)

    # 完善映射字典
    mode_map = {
        0: "📝 Text Mode (EEG + Text)",
        1: "🖼️ Image Mode (EEG + Image)",
        2: "🚀 Fusion Mode (Dynamic Fusion)",
        3: "Static Mode"
    }
    current_mode_desc = mode_map.get(pretrain_mode, f"Unknown Mode ({pretrain_mode})")

    # 完善短命名逻辑
    if pretrain_mode == 0:
        mode_str_short = "text"
    elif pretrain_mode == 1:
        mode_str_short = "img"
    elif pretrain_mode == 2:
        mode_str_short = "fusion"
    elif pretrain_mode == 3:
        mode_str_short = "static"
    else:
        mode_str_short = "unknown"
    print(mode_str_short)

    # 获取项目绝对根目录
    try:
        original_cwd = get_original_cwd()
    except:
        original_cwd = os.getcwd()

    # 打印运行状态
    print("\n" + "=" * 60)
    print(f"🚀 [Launch Configuration]")
    print(f"🔥 Pretrain Mode : {pretrain_mode} -> {current_mode_desc}")
    print(f"📂 Project Root  : {original_cwd}")
    print(f"📂 Hydra Log Dir : {os.getcwd()}")
    print("=" * 60 + "\n")

    # ================= [配置: 折数逻辑] =================
    if isinstance(cfg.train.valid_method, int):
        n_folds = cfg.train.valid_method
    elif cfg.train.valid_method == 'loo':
        n_folds = cfg.train.n_subs
    else:
        n_folds = 10

    n_per = round(cfg.data.n_subs / n_folds)

    # 👇 恢复正常的折数循环逻辑
    # fold_range = range(cfg.get("start_fold", 0), cfg.get("end_fold", n_folds))

    # 如果你想只跑第 9 折，取消下面这行的注释：
    fold_range = [0,1,2,3,4,5,6,7,8,9]

    print(f"👉 Running Folds: {list(fold_range)}")

    # ================= [配置: 预提取特征路径] =================
    # 🔧 请修改为你的特征文件路径。
    # 预提取的文本/图像特征可从项目 Release 页面下载，或使用 FACED/ 目录下的脚本自行提取。
    # 期望目录结构: features/text_timelen5_timestep2_1024_objective/*.npy
    #              features/image_features_clip_vit_centercrop_timelen5_timestep2/*.npy
    text_feat_dir = os.path.join(original_cwd, "features", "text_timelen5_timestep2_1024_objective")
    image_feat_dir = os.path.join(original_cwd, "features", "image_features_clip_vit_centercrop_timelen5_timestep2")

    # ===============================================

    # ===============================================

    for fold in fold_range:
        print(f"\n>>> Starting Fold: {fold} | Mode: {current_mode_desc} <<<")

        # ================= [Checkpoint 路径] =================
        # 🔧 Checkpoint 保存目录，可修改为任意路径
        cp_dir = os.path.join(original_cwd, "daest_cp")

        if fold == fold_range[0]:
            print(f"💾 Checkpoints will be saved to: {cp_dir}")

        os.makedirs(cp_dir, exist_ok=True)

        # 🎯 Proto_Acc: EEG 投影后与目标特征的检索准确率
        cp_monitor = "val/ZeroShot_Acc"
        es_monitor = "val/ZeroShot_Acc"
        print(f"📊 [Monitor] Using val/ZeroShot_Acc")

        checkpoint_callback = ModelCheckpoint(
            monitor=cp_monitor,
            mode="max",
            verbose=True,
            dirpath=cp_dir,
            filename=f'fold{fold}_' + '{epoch:02d}',
            save_top_k=1
        )

        earlyStopping_callback = EarlyStopping(
            monitor=es_monitor,
            mode="max",
            patience=cfg.train.patience,
            verbose=True
        )

        # 👇 [新增] 实例化自定义的权重打印回调
        weight_callback = EmotionWeightLogger()
        val_metrics_callback = ValMetricsLogger()

        # ================= [数据划分] =================
        if n_folds == 1:
            val_subs = []
        elif fold < n_folds - 1:
            val_subs = np.arange(n_per * fold, n_per * (fold + 1))
        else:
            val_subs = np.arange(n_per * fold, cfg.data.n_subs)

        train_subs = list(set(np.arange(cfg.data.n_subs)) - set(val_subs))
        if len(val_subs) == 1: val_subs = list(val_subs)

        print(f'   Train Subs: {len(train_subs)} subjects')
        print(f'   Val Subs:   {val_subs}')

        # 视频 ID 设置
        n_vids = 28 if cfg.data.dataset_name == 'FACED' else cfg.data.n_vids
        train_vids = np.arange(n_vids)
        val_vids = np.arange(n_vids)

        # ================= [初始化 DataModule] =================
        if cfg.data.dataset_name == 'FACED':
            dm = FACEDDataModule(
                load_dir=cfg.data.data_dir,
                save_dir=cfg.data.data_dir,
                timeLen=cfg.data.timeLen,
                timeStep=cfg.data.timeStep,
                train_subs=train_subs,
                val_subs=val_subs,
                train_vids=train_vids,
                val_vids=val_vids,
                n_session=cfg.data.n_session,
                fs=cfg.data.fs,
                n_chans=cfg.data.n_channs,
                n_subs=cfg.data.n_subs,
                n_vids=n_vids,
                n_class=cfg.data.n_class,
                loo=(cfg.train.valid_method == 'loo'),
                num_workers=cfg.train.num_workers,
                image_feat_dir=image_feat_dir,
                text_feat_dir=text_feat_dir,
                use_pretrain_sampler=True
            )
        else:
            dm = EEGDataModule(cfg.data, train_subs, val_subs, train_vids, val_vids,
                               cfg.train.valid_method == 'loo', cfg.train.num_workers)

        # ================= [同步 backbone 配置] =================
        from omegaconf import open_dict as od
        with od(cfg.model):
            cfg.model.proj_type = 'residual'
            cfg.model.use_ln_backbone = True
        # 其余配置 (use_modal_proj, probe_on_raw 等)
        # 由 ExtractorModel 从 cfg.train 直接读取, 不经过 cfg.model
        print(f"📐 proj_type=residual, probe_on_raw={cfg.train.get('probe_on_raw',False)}")

        # ================= [初始化模型] =================
        base_model = hydra.utils.instantiate(cfg.model)

        # 实例化 LightningModule
        Extractor = ExtractorModel(base_model, cfg.train)

        # ================= [Trainer 设置] =================
        trainer = pl.Trainer(
            logger=False,  # 关闭 TensorBoard 避免异步写盘崩溃
            # 👇 [关键] 必须把 weight_callback 加到这里的列表中
            callbacks=[checkpoint_callback, earlyStopping_callback, weight_callback, val_metrics_callback],
            max_epochs=cfg.train.max_epochs,
            min_epochs=cfg.train.min_epochs,
            accelerator='gpu',
            devices=cfg.train.gpus,
            limit_val_batches=0.0 if n_folds == 1 else 1.0,
            precision="bf16-mixed",
            gradient_clip_val=1.0,
            enable_progress_bar=True
        )

        # 开始训练
        trainer.fit(Extractor, dm)

        if cfg.train.iftest:
            print("🛑 Test mode enabled, stopping after first fold.")
            break


if __name__ == "__main__":
    train_ext()