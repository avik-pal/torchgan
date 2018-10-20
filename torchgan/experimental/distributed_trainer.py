import torch
import torchvision
import torch.nn as nn
from warnings import warn
from ..trainer.trainer import Trainer

__all__ = ['DistributedTrainer']

class DistributedTrainer(Trainer):
    def __init__(self, *args, gpu_ids=None, **kwargs):
        super(DistributedTrainer, self).__init__(*args, **kwargs)
        if self.device is not torch.device("cuda:0"):
            warn("`device` parameter is overwritten in Distributed Trainer. Refer to docs for choosing gpu")
        if gpu_ids is None:
            warn("Using all GPUs present. Specify `gpu_ids` to select which GPUs to use")
        self.gpu_ids = gpu_ids
        # Override the assignments made in Trainer
        self.generator = nn.DataParallel(self.generator, device_ids=self.gpu_ids)
        self.discriminator = nn.DataParallel(self.discriminator, device_ids=self.gpu_ids)
        # These reassignments are needed as the location of the models are being altered
        if "optimizer_generator_options" in kwargs:
            self.optimizer_generator = optimizer_generator(self.generator.parameters(),
                                                           **kwargs["optimizer_generator_options"])
        else:
            self.optimizer_generator = optimizer_generator(self.generator.parameters())
        if "optimizer_discriminator_options" in kwargs:
            self.optimizer_discriminator = optimizer_discriminator(self.discriminator.parameters(),
                                                **kwargs["optimizer_discriminator_options"])
        else:
            self.optimizer_discriminator = optimizer_discriminator(self.discriminator.parameters())
