from typing import List, Optional

import torch
from torch.optim import Optimizer
from pytorch_lightning.plugins import HorovodPlugin
from pytorch_lightning.utilities import _HOROVOD_AVAILABLE

if _HOROVOD_AVAILABLE:
    import horovod.torch as hvd
    hvd.init()


class CustomHorovodPlugin(HorovodPlugin):
    def __init__(self, parallel_devices: Optional[List[torch.device]] = None, backward_passes_per_step=1):
        super().__init__(parallel_devices=parallel_devices)
        self.backward_passes_per_step = backward_passes_per_step

    def _wrap_optimizers(self, optimizers: List[Optimizer]) -> List['hvd.DistributedOptimizer']:
        """Wraps optimizers to perform gradient aggregation via allreduce."""
        return [
            hvd.DistributedOptimizer(
                opt,
                named_parameters=HorovodPlugin._filter_named_parameters(self.lightning_module, opt),
                backward_passes_per_step=self.backward_passes_per_step
            )
            if 'horovod' not in str(opt.__class__)
            else opt
            for opt in optimizers
        ]
