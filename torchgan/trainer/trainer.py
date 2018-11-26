import os
import torch
import torchvision
from warnings import warn
from inspect import signature, _empty
from operator import itemgetter
from ..models.model import Generator, Discriminator
from ..losses.loss import GeneratorLoss, DiscriminatorLoss
from ..logging.logger import Logger

__all__ = ['Trainer']

class Trainer(object):
    r"""Base class for all Trainers for various GANs.

    Features provided by this Base Trainer are:

    - Loss and Metrics Logging
    - Generating Image Samples
    - Logging using Tensorboard
    - Saving models at the end of every epoch and loading of previously saved models
    - Highly flexible and allows changing hyperparameters by simply adjusting the keyword arguments.
    - Custom `train_ops` by mapping the function signature with the values stored in the object

    Most of the functionalities provided by the Trainer are flexible enough and can be customized by
    simply passing different arguments. You can train anything from a simple DCGAN to complex CycleGANs
    without ever having to subclass this `Trainer`.

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
        metrics_list (list, optional): List of Metric Functions that need to be logged. For a list of pre-defined
                       metrics look at :mod:`torchgan.metrics`. All losses in the list must be a subclass of
                       `EvaluationMetric`.
        schedulers (dict, optional): Schedulers can either be from Pytorch or can be a custom scheduler as long as
                       it strictly follows that of Pytorch. If your optimizer is named `optimizer_my_new_model` then
                       the corresponding scheduler must be named `scheduler_optimizer_my_new_model`.
        device (torch.device, optional): Device in which the operation is to be carried out. If you are using a
                       CPU machine make sure that you change it for proper functioning.
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
        tensorboard (bool, optional): If `True`, tensorboard logs will be generated in the `runs` directory.
        log_dir (str, optional): The directory for logging tensorboard.
        test_noise (torch.Tensor, optional): If provided then it will be used as the noise for image sampling.
        no_print (bool, optional): Set it to `True` if you don't want to see any result printed to the console.

    Any other argument that you need to store in the object can be simply passed via keyword arguments.

    Example:
        >>> dcgan = Trainer(
                    {"generator": {"name": DCGANGenerator, "args": {"out_channels": 1, "step_channels": 16}},
                     "discriminator": {"name": DCGANDiscriminator, "args": {"in_channels": 1, "step_channels": 16}}},
                    {"optimizer_generator": {"name": Adam, "args": {"lr": 0.0002, "betas": (0.5, 0.999)}},
                     "optimizer_discriminator": {"name": Adam, "args": {"lr": 0.0002, "betas": (0.5, 0.999)}}},
                    [MinimaxGeneratorLoss(), MinimaxDiscriminatorLoss()],
                    sample_size=64, epochs=20)
    """
    def __init__(self, models, optimizers, losses_list, metrics_list=None, schedulers=None,
                 device=torch.device("cuda:0"), ncritic=None, batch_size=128, epochs=5,
                 sample_size=8, checkpoints="./model/gan", retain_checkpoints=5, recon="./images",
                 tensorboard=True, log_dir=None, test_noise=None, no_print=False,**kwargs):
        self.device = device
        self.model_names = []
        for key, val in models.items():
            self.model_names.append(key)
            if "args" in val:
                setattr(self, key, (val["name"](**val["args"])).to(self.device))
            else:
                setattr(self, key, (val["name"]()).to(self.device))
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
        for loss in losses_list:
            self.losses[type(loss).__name__] = loss
        if metrics_list is None:
            self.metrics = None
        else:
            for metric in metrics_list:
                self.metrics[type(metric).__name__] = metric
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.epochs = epochs
        self.checkpoints = checkpoints
        self.retain_checkpoints = retain_checkpoints
        self.recon = recon

        # Not needed but we need to store this to avoid errors. Also makes life simpler
        self.noise = None
        self.real_inputs = None
        self.labels = None

        self.loss_information = {
            'generator_losses': 0.0,
            'discriminator_losses': 0.0,
            'generator_iters': 0,
            'discriminator_iters': 0,
        }
        self.ncritic = ncritic
        self.start_epoch = 0
        self.last_retained_checkpoint = 0
        for key, val in kwargs.items():
            if key in self.__dict__:
                warn("Overiding the default value of {} from {} to {}".format(key, getattr(self, key), val))
            setattr(self, key, val)

        self.logger = Logger(self, losses_list, metrics_list, tensorboard=tensorboard,
                             log_dir=log_dir, nrow=nrow, test_noise=test_noise, no_print=no_print)

        os.makedirs(self.checkpoints.rsplit("/", 1)[0], exist_ok=True)
        os.makedirs(self.recon, exist_ok=True)

    def save_model(self, epoch, save_items=None):
        r"""Function saves the model and some necessary information along with it. List of items
        stored for future reference:

        - Epoch
        - Model States
        - Optimizer States
        - Loss Information
        - Loss Objects
        - Metric Objects
        - Loss Logs
        - Metric Logs

        Args:
            epoch (int, optional): Epoch Number at which the model is being saved
            save_items (str, list, optional): Pass the variable name of any other item you want to save.
                                              The item must be present in the `__dict__` else training
                                              will come to an abrupt end.
        """
        if self.last_retained_checkpoint == self.retain_checkpoints:
            self.last_retained_checkpoint = 0
        save_path = self.checkpoints + str(self.last_retained_checkpoint) + '.model'
        self.last_retained_checkpoint += 1
        print("Saving Model at '{}'".format(save_path))
        model = {
            'epoch': epoch + 1,
            'loss_information': self.loss_information,
            'loss_objects': self.losses,
            'metric_objects': self.metrics,
            'loss_logs': self.loss_logs,
            'metric_logs': self.metric_logs
        }
        for save_item in self.model_names + self.optimizer_names:
            model.update({save_item: (getattr(self, save_item)).state_dict()})
        if save_items is not None:
            if type(save_items) is list:
                for itms in save_items:
                    model.update({itms: getattr(self, itms)})
            else:
                model.update({save_items: getattr(self, save_items)})
        torch.save(model, save_path)

    def load_model(self, load_path="", load_items=None):
        r"""Function to load the model and some necessary information along with it. List of items
        loaded:

        - Epoch
        - Model States
        - Optimizer States
        - Loss Information
        - Loss Objects
        - Metric Objects
        - Loss Logs
        - Metric Logs

        Args:
            load_path (string, optional): Path from which the model is to be loaded.
            load_items (str, list, optional): Pass the variable name of any other item you want to load.
                                              If the item cannot be found then a warning will be thrown
                                              and model will start to train from scratch. So make sure
                                              that item was saved.
        """
        if load_path == "":
            load_path = self.checkpoints + str(self.last_retained_checkpoint) + '.model'
        print("Loading Model From '{}'".format(load_path))
        try:
            checkpoint = torch.load(load_path)
            self.start_epoch = checkpoint['epoch']
            self.losses = checkpoint['loss_objects']
            self.metrics = checkpoint['metric_objects']
            self.loss_information = checkpoint['loss_information']
            self.loss_logs = checkpoint['loss_logs']
            self.metric_logs = checkpoint['metric_logs']
            for load_item in self.model_names + self.optimizer_names:
                getattr(self, load_item).load_state_dict(checkpoint[load_item])
            if load_items is not None:
                if type(load_items) is list:
                    for itms in load_items:
                        setattr(self, itms, checkpoint['itms'])
                else:
                    setattr(self, load_items, checkpoint['load_items'])
        except:
            warn("Model could not be loaded from {}. Training from Scratch".format(load_path))

    def set_arg_maps(self, mappings):
        r"""Helper function to allow custom parameter names in Loss and Metric Functions.
        Also allows storing of buffers.

        Args:
            mappings (list, tuple): It must be a tuple or a list of tuples, where the first variable
                                    name is the name needed by the function and the next variable name is
                                    the name of the variable already stored in the object.

        Example:
            >>> #If your loss function needs a parameter `gen` which is stored as `generator` in the trainer.
            >>> trainer.set_arg_maps(("gen", "generator"))
        """
        if type(mappings) is list:
            for mapping in mappings:
                setattr(self, mapping[0], mapping[1])
        else:
            setattr(self, mappings[0], mappings[1])

    def _get_argument_maps(self, func):
        r"""Extracts the signature of the `func`. Then it returns the list of arguments that
        are present in the object and need to be mapped and passed to the `func` when calling it.

        Args:
            func (Function): Function whose argument map is to be generated

        Returns:
            List of arguments that need to be fed into the function. It contains all the positional
            arguments and keyword arguments that are stored in the object. If any of the required
            arguments are not present an error is thrown.
        """
        sig = signature(func)
        args = [p.name for p in sig.parameters.values() if p.default is _empty\
                    and p.name != 'kwargs' and p.name != 'args']
        for arg in args:
            if arg not in self.__dict__:
                raise Exception("Argument : {} not present. If the value is stored with some other\
                                 name use the function `set_arg_maps`".format(arg))
        for arg in [p.name for p in sig.parameters.values() if p.default is not _empty]:
            if arg in self.__dict__:
                args.append(arg)
        return args

    def _store_metric_maps(self):
        r"""Creates a mapping between the metrics and the arguments from the object that need to be
        passed to it.
        """
        if self.metrics is not None:
            self.metric_arg_maps = {}
            for name, metric in self.metrics.items():
                self.metric_arg_maps[name] = self._get_argument_maps(metric.metric_ops)

    def _store_loss_maps(self):
        r"""Creates a mapping between the losses and the arguments from the object that need to be
        passed to it.
        """
        self.loss_arg_maps = {}
        for name, loss in self.losses.items():
            self.loss_arg_maps[name] = self._get_argument_maps(loss.train_ops)

    def _get_arguments(self, arg_map):
        r"""Get the argument values from the object and create a dictionary.

        Args:
            arg_map (list): A list of arguments that is generated by `_get_argument_maps`.

        Returns:
            A dictionary mapping the argument name to the value of the argument.
        """
        return dict(zip(arg_map, itemgetter(*arg_map)(self.__dict__)))

    def train_iter_custom(self):
        r"""Function that needs to be extended if `train_iter` is to be modified. Use this function
        to perform any sort of initialization that need to be done at the beginning of any train
        iteration. Refer the model zoo and example docs for more details on how to write this function.
        """
        pass

    # TODO(avik-pal): Clean up this function and avoid returning values
    def train_iter(self):
        r"""Calls the train_ops of the loss functions. This is the core function of the Trainer. In most
        cases you will never have the need to extend this function. In extreme cases simply extend
        `train_iter_custom`.

        Returns:
            An NTuple of the generator loss, discriminator loss, times the generator was trained and the number
            of times the discriminator was trained.
        """
        self.train_iter_custom()
        ldis, lgen, dis_iter, gen_iter = 0.0, 0.0, 0, 0
        for name, loss in self.losses.items():
            loss_logs = self.logger.get_loss_viz()
            if isinstance(loss, GeneratorLoss) and isinstance(loss, DiscriminatorLoss):
                cur_loss = loss.train_ops(**self._get_arguments(self.loss_arg_maps[name]))
                loss_logs.logs[name].append(cur_loss)
                if type(cur_loss) is tuple:
                    lgen, ldis, gen_iter, dis_iter = lgen + cur_loss[0], ldis + cur_loss[1],\
                        gen_iter + 1, dis_iter + 1
            elif isinstance(loss, GeneratorLoss):
                if self.ncritic is None or\
                   self.loss_information["discriminator_iters"] % self.ncritic == 0:
                    cur_loss = loss.train_ops(**self._get_arguments(self.loss_arg_maps[name]))
                    loss_logs.logs[name].append(cur_loss)
                    lgen, gen_iter = lgen + cur_loss, gen_iter + 1
            elif isinstance(loss, DiscriminatorLoss):
                cur_loss = loss.train_ops(**self._get_arguments(self.loss_arg_maps[name]))
                loss_logs.logs[name].append(cur_loss)
                ldis, dis_iter = ldis + cur_loss, dis_iter + 1
        return lgen, ldis, gen_iter, dis_iter

    def eval_ops(self, epoch, **kwargs):
        r"""Runs all evaluation operations at the end of every epoch. It calls all the metric functions that
        are passed to the Trainer. Also calls the image sampler.

        Args:
            epoch (int): Current epoch at which the sampler is being called.
        """
        if self.metrics is not None:
            for name, metric in self.metrics.items():
                metric_logs = self.logger.get_metric_viz()
                metric_logs.logs[name].append(metric.metric_ops(\
                    **self._get_arguments(self.metric_arg_maps[name])))

    def optim_ops(self):
        r"""Runs all the schedulers at the end of every epoch.
        """
        for scheduler in self.schedulers:
            scheduler.step()

    def train(self, data_loader, **kwargs):
        r"""Uses the information passed by the user while creating the object and trains the model.
        It iterates over the epochs and the Data and calls the functions for training the models and
        logging the required variables. You should never try to extend this function. It is too delicate
        and changing it affects every other function present in this Trainer class.

        Args:
            data_loader (torch.DataLoader): A DataLoader for the trainer to iterate over and train the
                                            models.
        """
        for epoch in range(self.start_epoch, self.epochs):

            for model in self.model_names:
                getattr(self, model).train()

            for data in data_loader:
                if type(data) is tuple or type(data) is list:
                    self.real_inputs = data[0].to(self.device)
                    self.labels = data[1].to(self.device)
                elif type(data) is torch.Tensor:
                    self.real_inputs = data.to(self.device)
                else:
                    self.real_inputs = data

                lgen, ldis, gen_iter, dis_iter = self.train_iter()
                self.loss_information['generator_losses'] += lgen
                self.loss_information['discriminator_losses'] += ldis
                self.loss_information['generator_iters'] += gen_iter
                self.loss_information['discriminator_iters'] += dis_iter

                self.logger.run_mid_epoch(self)

            if "save_items" in kwargs:
                self.save_model(epoch, kwargs["save_items"])
            else:
                self.save_model(epoch)

            for model in self.model_names:
                getattr(self, model).eval()

            self.eval_ops(epoch, **kwargs)
            self.logger.run_end_epoch(self, epoch)
            self.optim_ops()

        print("Training of the Model is Complete")

    def complete(self, **kwargs):
        r"""Marks the end of training. It saves the final model and turns off the
        logger.
        """
        if "save_items" in kwargs:
            self.save_model(-1, kwargs["save_items"])
        else:
            self.save_model(-1)
        self.logger.close()

    def __call__(self, data_loader, **kwargs):
        self._store_loss_maps()
        self._store_metric_maps()
        self.batch_size = data_loader.batch_size
        self.train(data_loader, **kwargs)
