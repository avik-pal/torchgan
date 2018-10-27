import torch
import torchvision
import torch.nn as nn
from warnings import warn
from ..trainer.trainer import Trainer

__all__ = ['DistributedTrainer']

class DistributedTrainer(Trainer):
    r"""Derived from the Trainer class. To be used when multiple GPUs are available for training.
    There are certain issues when using Distributed Training. Refer to `this page
    <https://pytorch.org/docs/stable/nn.html#dataparallel-layers-multi-gpu-distributed>` for
    details on them.

    Features provided by this Base Trainer are:

    - Loss and Metrics Logging
    - Generating Image Samples
    - Logging using Tensorboard
    - Saving models at the end of every epoch and loading of previously saved models
    - Highly flexible and allows changing hyperparameters by simply adjusting the keyword arguments.
    - Custom `train_ops` by mapping the function signature with the values stored in the object

    Args:
        models (dict): A dictionary containing a mapping between the variable name, storing the `generator`,
                       `discriminator` and any other model that you might want to define, with the function and
                       arguments that are needed to construct the model. Refer to the examples to see how to
                       define complex models using this API.
        optimizers (dict): Contains a mapping between the variable name of the optimizer and the function and arguments
                       needed to construct the optimizer. Naming convention that is to be used for the proper
                       functioning of the optimizer: If your model is named `my_new_model` then the optimizer
                       corresponding to that model must be named `optimizer_my_new_model`. Following any other naming
                       convention will simply throw an error.
        losses_list (list): A list of the Loss Functions that need to be minimized. For a list of pre-defined losses
                       look at :mod:`torchgan.losses`. All losses in the list must be a subclass of atleast
                       `GeneratorLoss` or `DiscriminatorLoss`.
        gpu_ids (list): Numbers corresponding to the GPUs to be used for training.
        metrics_list (list, optional): List of Metric Functions that need to be logged. For a list of pre-defined
                       metrics look at :mod:`torchgan.metrics`. All losses in the list must be a subclass of
                       `EvaluationMetric`.
        schedulers (dict, optional): Schedulers can either be from Pytorch or can be a custom scheduler as long as
                       it strictly follows that of Pytorch. If your optimizer is named `optimizer_my_new_model` then
                       the corresponding scheduler must be named `scheduler_optimizer_my_new_model`.
        ncritic (int, optional): Setting it to a value will make the discriminator train that many times more than
                       the generator.
        batch_size (int, optional): Batch Size for feeding into the discriminator.
        sample_size (int, optional): Total number of images to be generated at the end of an epoch for logging
                       purposes.
        epochs (int, optional): Total number of epochs for which the models are to be trained.
        checkpoints (str, optional): Path where the models are to be saved. The naming convention is if checkpoints
                       is `./model/gan` then models are saved as `./model/gan0.model` and so on. Make sure that the
                       `model` directory exists from before.
        retain_checkpoints (int, optional): Total number of checkpoints that should be retained. For example,
                       if the value is set to 3, we save at most 3 models and start rewriting the models after that.
        recon (str, optional): Directory where the sampled images are saved. Make sure the directory exists from
                       beforehand.
        log_tensorboard (bool, optional): If `True`, tensorboard logs will be generated in the `runs` directory.
        test_noise (torch.Tensor, optional): If provided then it will be used as the noise for image sampling.

    Any other argument that you need to store in the object can be simply passed via keyword arguments.

    Example:
        >>> dcgan = DistributedTrainer(
                    {"generator": {"name": DCGANGenerator, "args": {"out_channels": 1, "step_channels": 16}},
                     "discriminator": {"name": DCGANDiscriminator, "args": {"in_channels": 1, "step_channels": 16}}},
                    {"optimizer_generator": {"name": Adam, "args": {"lr": 0.0002, "betas": (0.5, 0.999)}},
                     "optimizer_discriminator": {"name": Adam, "args": {"lr": 0.0002, "betas": (0.5, 0.999)}}},
                    [MinimaxGeneratorLoss(), MinimaxDiscriminatorLoss()], gpu_ids=[0,1,2]
                    sample_size=64, epochs=20)
        >>> # This simple code allows training to be performed on 3 GPUs simultaneously
    """
    def __init__(self, models, optimizers, losses_list, gpu_ids, metrics_list=None,
                 schedulers=None, ncritic=None, batch_size=128,
                 sample_size=8, epochs=5, checkpoints="./model/gan", retain_checkpoints=5,
                 recon="./images", log_tensorboard=True, test_noise=None, **kwargs):
        self.gpu_ids = gpu_ids
        self.model_names = []
        for key, val in models.items():
            self.model_names.append(key)
            if "args" in val:
                setattr(self, key, nn.DataParallel(val["name"](**val["args"]), device_ids=self.gpu_ids))
            else:
                setattr(self, key, nn.DataParallel(val["name"](), device_ids=self.gpu_ids))
        self.optimizer_names = []
        for key, val in optimizers.items():
            self.optimizer_names.append(key)
            model = getattr(self, key.split("_", 1)[1])
            if "args" in val:
                setattr(self, key, val["name"](model.parameters(), **val["args"]))
            else:
                setattr(self, key, val["name"](model.parameters()))
        self.schedulers = []
        if schedulers is not None:
            for key, val in schedulers.items():
                opt = getattr(self, key.split("_", 1)[1])
                if "args" in val:
                    self.schedulers.append(val["name"](opt, **val["args"]))
                else:
                    self.schedulers.append(val["name"](opt))
        self.losses = {}
        self.loss_logs = {}
        for loss in losses_list:
            name = type(loss).__name__
            self.loss_logs[name] = []
            self.losses[name] = loss
        if metrics_list is None:
            self.metrics = None
            self.metric_logs = None
        else:
            self.metric_logs = {}
            self.metrics = {}
            for metric in metrics_list:
                name = type(metric).__name__
                self.metric_logs[name] = []
                self.metrics[name] = metric
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.epochs = epochs
        self.checkpoints = checkpoints
        self.retain_checkpoints = retain_checkpoints
        self.recon = recon
        # NOTE(avik-pal): We should also check if the length of the list is equal to the number of
        #                 generators
        if test_noise is None or type(test_noise) is not list:
            self.test_noise = []
            for model in self.model_names:
                if isinstance(getattr(self, model), Generator):
                    self.test_noise.append(torch.randn(self.sample_size,
                                                getattr(self, model).encoding_dims,
                                                device=self.device) if test_noise is None else test_noise)
        # Not needed but we need to store this to avoid errors. Also makes life simpler
        self.noise = torch.randn(1)
        self.real_inputs = torch.randn(1)
        self.labels = torch.randn(1)

        self.loss_information = {
            'generator_losses': 0.0,
            'discriminator_losses': 0.0,
            'generator_iters': 0,
            'discriminator_iters': 0,
        }
        self.ncritic = ncritic
        self.start_epoch = 0
        self.last_retained_checkpoint = 0
        self.log_tensorboard = log_tensorboard
        if self.log_tensorboard:
            self.tensorboard_information = {
                "step": 0,
                "repeat_step": 4,
                "repeats": 1
            }
        self.nrow = 8
        for key, val in kwargs.items():
            if key in self.__dict__():
                warn("Overiding the default value of {} from {} to {}".format(key, getattr(self, key), val))
            setattr(self, key, val)
