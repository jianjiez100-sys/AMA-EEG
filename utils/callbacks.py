# utils/callbacks.py
from pytorch_lightning.callbacks import EarlyStopping


class WarmupEarlyStopping(EarlyStopping):
    """
    继承自 PyTorch Lightning 的 EarlyStopping。
    功能：在指定的 warmup_epochs 轮次之前，强制不进行早停检查。
    """

    def __init__(self, warmup_epochs: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.warmup_epochs = warmup_epochs

    def on_validation_end(self, trainer, pl_module):
        # 1. 如果当前轮数还处于预热期，直接跳过，什么都不做
        if trainer.current_epoch < self.warmup_epochs:
            return

            # 2. 否则，执行父类 (官方逻辑) 的检查
        super().on_validation_end(trainer, pl_module)