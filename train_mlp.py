import hydra
from omegaconf import DictConfig
from model.models import simpleNN3
import numpy as np
import os
from data.dataset import PDataset
from model.pl_models import MLPModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from torch.utils.data import DataLoader
import torch
import logging
from hydra.utils import get_original_cwd, to_absolute_path
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
# --- [新增] 导入混淆矩阵 ---
from sklearn.metrics import confusion_matrix

# ------------------------

log = logging.getLogger(__name__)


# ==========================================
# ✅ 3. 新增 MetricsCallback 类
#    作用：在训练进度条中实时显示 F1 和 Kappa
# ==========================================
class MetricsCallback(Callback):
    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.val_preds = []
        self.val_targets = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        with torch.no_grad():
            logits = pl_module(x)
            preds = torch.argmax(logits, dim=1)

        self.val_preds.append(preds.cpu())
        self.val_targets.append(y.cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.val_preds:
            return

        all_preds = torch.cat(self.val_preds).numpy()
        all_targets = torch.cat(self.val_targets).numpy()

        # 计算指标：二分类用 binary，多分类用 macro
        avg_method = 'binary' if self.out_dim == 2 else 'macro'

        f1 = f1_score(all_targets, all_preds, average=avg_method)
        kappa = cohen_kappa_score(all_targets, all_preds)

        # 记录到进度条
        pl_module.log('mlp/val/f1', f1, prog_bar=True)
        pl_module.log('mlp/val/kappa', kappa, prog_bar=True)

        # 清空列表
        self.val_preds = []
        self.val_targets = []


@hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
def train_mlp(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cls_mode = int(cfg.data.n_class)
    feature_dir = to_absolute_path(cfg.ext_fea.output_dir)
    feature_suffix = str(cfg.ext_fea.mode)
    n_folds = cfg.data.n_subs if cfg.train.valid_method == 'loo' else int(cfg.train.valid_method)
    end_fold = n_folds if cfg.get('end_fold') is None else min(int(cfg.end_fold), n_folds)
    target_folds = list(range(int(cfg.get('start_fold', 0)), end_fold))
    if cfg.train.iftest:
        target_folds = target_folds[:1]
    mode_str = f"{cls_mode}-Class {cfg.data.dataset_name}"
    log.info("=" * 60)
    log.info(f"🚀 Current Task Mode: {mode_str}")
    log.info(f"📂 Feature Path: {feature_dir}")
    log.info(f"🎯 Target Folds: {target_folds}")
    log.info("=" * 60)

    cfg.mlp.out_dim = cls_mode

    # --- [新增] 根据分类模式选择标签列表 ---
    emotion_list = list(cfg.data.class_names)
    # -------------------------------------

    n_per = round(cfg.data.n_subs / n_folds) if n_folds > 0 else 1

    # ✅ 4. 修改存储结构，记录 Acc, F1, Kappa
    final_metrics = {
        "acc": [],
        "f1": [],
        "kappa": [],
        # --- [新增] 扩充数据记录容器 ---
        "per_class_acc": [],
        "all_preds_global": [],
        "all_targets_global": []
        # -----------------------------
    }

    root_dir = get_original_cwd()

    for fold in target_folds:
        cp_dir = os.path.join(root_dir, 'checkpoints_mlp', cfg.data.dataset_name, f'cls{cls_mode}_r{cfg.log.run}')
        os.makedirs(cp_dir, exist_ok=True)

        cp_monitor = None if n_folds == 1 else "mlp/val/acc"
        es_monitor = "mlp/train/acc" if n_folds == 1 else "mlp/val/acc"
        filename_fmt = f'fold{fold}_acc=' + '{mlp/val/acc:.4f}'

        checkpoint_callback = ModelCheckpoint(
            monitor=cp_monitor,
            verbose=True,
            mode="max",
            dirpath=cp_dir,
            filename=filename_fmt,
            save_top_k=1
        )

        earlyStopping_callback = EarlyStopping(monitor=es_monitor, mode="max", patience=cfg.mlp.patience)

        # ✅ 5. 实例化 Callback
        metrics_callback = MetricsCallback(out_dim=cls_mode)

        log.info(f"\n🚀 Training Fold: {fold}")

        # 数据划分
        if n_folds == 1:
            val_subs = []
        elif fold < n_folds - 1:
            val_subs = np.arange(n_per * fold, n_per * (fold + 1))
        else:
            val_subs = np.arange(n_per * fold, cfg.data.n_subs)
        train_subs = list(set(np.arange(cfg.data.n_subs)) - set(val_subs))

        # 加载特征
        save_dir = feature_dir
        patterns = [
            os.path.join(save_dir, f"*_f{fold}_fea_{feature_suffix}.npy"),
            os.path.join(save_dir, f"fea_f{fold}_{feature_suffix}.npy"),
        ]
        found_files = []
        for pattern in patterns:
            found_files.extend(glob.glob(pattern))

        if not found_files:
            log.error(f"❌ Feature file NOT found for fold {fold}")
            continue

        save_path = found_files[0]
        try:
            data2 = np.load(save_path)
            log.info(f'✅ Feature loaded: {os.path.basename(save_path)}')
        except Exception as e:
            log.error(f"Failed to load feature: {e}")
            continue

        # ==========================================
        # 🚨 基础数据清洗 (NaN / Inf 处理)
        # ==========================================
        if np.isnan(data2).any() or np.isinf(data2).any():
            log.warning(f"⚠️ Fold {fold}: Data contains NaN or Inf. Cleaning...")
            # 遇到 inf 时，为了配合后续更宽松的 Pre-clip，这里也放宽替换值为 10.0
            data2 = np.nan_to_num(data2, nan=0.0, posinf=1000.0, neginf=-1000.0)

        n_subs = cfg.data.n_subs
        fea_dim = data2.shape[-1]
        try:
            data2 = data2.reshape(n_subs, -1, fea_dim)
        except ValueError:
            log.error(f"Reshape failed! {data2.shape}")
            continue

        # 加载标签
        label_path = os.path.join(save_dir, 'onesub_label2.npy')
        onesub_label2 = np.load(label_path)

        # 准备数据分布
        labels2_train = np.tile(onesub_label2, len(train_subs))
        labels2_val = np.tile(onesub_label2, len(val_subs))

        # ==========================================
        # 🚀 [核心修改] “双重保险”标准化逻辑
        # ==========================================
        train_data_flat = data2[train_subs].reshape(-1, fea_dim)
        val_data_flat = data2[val_subs].reshape(-1, fea_dim)

        log.info(f"   Standardizing features for Fold {fold}...")


        # 🧠 标准化：此时 scaler 能够安全地学到正常数据的分布规律
        scaler = StandardScaler()
        train_data_flat = scaler.fit_transform(train_data_flat)
        if val_data_flat.shape[0] > 0:
            val_data_flat = scaler.transform(val_data_flat)

        # 🚨 第二道保险 (Post-Clip)：掐灭除零爆炸
        # 防止那些原本几乎全是 0 的死特征，因为除以极小标准差而放大到成百上千
        POST_CLIP_LIMIT = 3.0
        train_data_flat = np.clip(train_data_flat, -POST_CLIP_LIMIT, POST_CLIP_LIMIT)
        if val_data_flat.shape[0] > 0:
            val_data_flat = np.clip(val_data_flat, -POST_CLIP_LIMIT, POST_CLIP_LIMIT)
        log.info(
            f"   ✂️ Post-clip applied at [{-POST_CLIP_LIMIT}, {POST_CLIP_LIMIT}] to prevent division-by-zero explosions.")

        # ==========================================
        # 🚀 [可选] PCA 降维: 1024 → 64/128
        # ==========================================
        fea_dim_orig = fea_dim
        if cfg.mlp.get('use_pca', False):
            from sklearn.decomposition import PCA
            pca_dim = cfg.mlp.get('pca_dim', 64)
            pca = PCA(n_components=pca_dim)
            train_data_flat = pca.fit_transform(train_data_flat)
            if val_data_flat.shape[0] > 0:
                val_data_flat = pca.transform(val_data_flat)
            fea_dim = pca_dim
            explained = pca.explained_variance_ratio_.sum()
            log.info(f"   📐 PCA: {fea_dim_orig} → {pca_dim}, explained variance: {explained:.4f} ({explained*100:.1f}%)")
        # ==========================================

        log.info(f"   Train Samples: {train_data_flat.shape[0]}, Val Samples: {val_data_flat.shape[0]}")
        # ==========================================

        trainset2 = PDataset(train_data_flat, labels2_train)
        valset2 = PDataset(val_data_flat, labels2_val)

        trainLoader = DataLoader(trainset2, batch_size=cfg.mlp.batch_size, shuffle=True,
                                 num_workers=cfg.mlp.num_workers)
        valLoader = DataLoader(valset2, batch_size=cfg.mlp.batch_size, shuffle=False,
                               num_workers=cfg.mlp.num_workers)

        # 模型实例化
        model_mlp = simpleNN3(
            inp_dim=fea_dim,
            hidden_dim=cfg.mlp.hidden_dim,
            out_dim=cfg.mlp.out_dim,
            dropout=cfg.mlp.dropout,
            bn=cfg.mlp.get('bn', 'no')
        )
        predictor = MLPModel(model_mlp, cfg.mlp)

        limit_val_batches = 0.0 if n_folds == 1 else 1.0

        trainer = pl.Trainer(logger=False,
                             # ✅ 6. 将 metrics_callback 加入 Trainer
                             callbacks=[checkpoint_callback, earlyStopping_callback, metrics_callback],
                             max_epochs=cfg.mlp.max_epochs,
                             min_epochs=cfg.mlp.min_epochs,
                             accelerator=cfg.train.accelerator,
                             devices=1 if cfg.train.accelerator == 'cpu' else cfg.mlp.gpus,
                             limit_val_batches=limit_val_batches,
                             enable_progress_bar=True)

        trainer.fit(predictor, trainLoader, valLoader)

        # ✅ 7. 训练结束后，重新加载最佳模型计算该折的最终指标
        if cfg.train.valid_method != 1 and checkpoint_callback.best_model_path:
            best_model = MLPModel.load_from_checkpoint(
                checkpoint_callback.best_model_path,
                model=model_mlp,
                cfg=cfg.mlp
            )
            best_model.eval()
            best_model.to("cuda" if torch.cuda.is_available() else "cpu")

            all_preds = []
            all_targets = []

            with torch.no_grad():
                for batch in valLoader:
                    x, y = batch
                    x = x.to(best_model.device)
                    logits = best_model(x)
                    preds = torch.argmax(logits, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(y.numpy())

            # 计算各项指标
            fold_acc = accuracy_score(all_targets, all_preds)
            avg_method = 'binary' if cls_mode == 2 else 'macro'
            fold_f1 = f1_score(all_targets, all_preds, average=avg_method)
            fold_kappa = cohen_kappa_score(all_targets, all_preds)

            final_metrics["acc"].append(fold_acc)
            final_metrics["f1"].append(fold_f1)
            final_metrics["kappa"].append(fold_kappa)

            # --- [新增] 折内类别级准确率计算与记录 ---
            cm = confusion_matrix(all_targets, all_preds, labels=range(cls_mode))
            with np.errstate(divide='ignore', invalid='ignore'):
                per_class = cm.diagonal() / cm.sum(axis=1)
                per_class = np.nan_to_num(per_class)

            final_metrics["per_class_acc"].append(per_class)
            final_metrics["all_preds_global"].extend(all_preds)
            final_metrics["all_targets_global"].extend(all_targets)
            # ---------------------------------------

            log.info(f"✨ Fold {fold} Result: Acc={fold_acc * 100:.2f}%, F1={fold_f1:.4f}, Kappa={fold_kappa:.4f}")

            # --- [新增] 在本折结束时打印情感准确率 ---
            log.info(f"📊 Fold {fold} Per-Class Accuracy:")
            class_info = " | ".join([f"{emotion_list[i]}: {per_class[i] * 100:.1f}%" for i in range(cls_mode)])
            log.info(f"   {class_info}")
            # ----------------------------------------

        if cfg.train.iftest:
            break

    # ✅ 8. 最终汇总输出
    if cfg.train.valid_method != 1 and len(final_metrics["acc"]) > 0:
        log.info("\n" + "=" * 60)
        log.info(f" 🏆 FINAL RESULTS (Mode: {cls_mode}-Class)")
        log.info("=" * 60)

        header = f"{'Sub':<5} | {'Acc (%)':<10} | {'F1 Score':<10} | {'Kappa':<10}"
        log.info(header)
        log.info("-" * len(header))

        for i in range(len(final_metrics["acc"])):
            real_fold = target_folds[i]
            log.info(
                f"{real_fold:<5} | {final_metrics['acc'][i] * 100:.2f}       | {final_metrics['f1'][i]:.4f}       | {final_metrics['kappa'][i]:.4f}")

        log.info("-" * len(header))

        mean_acc = np.mean(final_metrics["acc"]) * 100
        std_acc = np.std(final_metrics["acc"]) * 100
        mean_f1 = np.mean(final_metrics["f1"])
        std_f1 = np.std(final_metrics["f1"])
        mean_kappa = np.mean(final_metrics["kappa"])
        std_kappa = np.std(final_metrics["kappa"])

        log.info(f"AVG ACC   : {mean_acc:.2f}% ± {std_acc:.2f}%")
        log.info(f"AVG F1    : {mean_f1:.4f}  ± {std_f1:.4f}")
        log.info(f"AVG KAPPA : {mean_kappa:.4f}  ± {std_kappa:.4f}")

        # --- [新增] 全局类别总结和混淆矩阵输出 ---
        log.info("\n" + "=" * 60)
        log.info("📈 Average Performance Per Emotion Class:")
        mean_per_class = np.mean(final_metrics["per_class_acc"], axis=0)
        std_per_class = np.std(final_metrics["per_class_acc"], axis=0)

        header_cls = f"{'Emotion':<15} | {'Mean Acc (%)':<15} | {'Std (%)':<10}"
        log.info(header_cls)
        log.info("-" * len(header_cls))
        for i in range(cls_mode):
            log.info(f"{emotion_list[i]:<15} | {mean_per_class[i] * 100:<15.2f} | {std_per_class[i] * 100:<10.2f}")

        log.info("\n🧱 Global Confusion Matrix (Aggregated All Folds):")
        global_cm = confusion_matrix(final_metrics["all_targets_global"], final_metrics["all_preds_global"])

        top_line = "      " + "".join([f"[{i:^3}]" for i in range(cls_mode)])
        log.info(top_line)
        for i, row in enumerate(global_cm):
            row_str = f"[{i:^3}] " + "".join([f"{val:^5}" for val in row])
            log.info(f"{row_str}  <- {emotion_list[i]}")

        log.info("=" * 60)
        # -----------------------------------------


if __name__ == '__main__':
    train_mlp()
