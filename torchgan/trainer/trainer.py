import torch
import torchvision
from warnings import warn
from inspect import signature
from operator import itemgetter
from datetime import datetime
from tensorboardX import SummaryWriter
from ..losses.loss import GeneratorLoss, DiscriminatorLoss

__all__ = ['Trainer']

class Trainer(object):
    # NOTE(avik-pal): It is good practice to name models as "generator" and "discriminator" unless its a
    #                 multi-GAN model
    def __init__(self, models, optimizers, losses_list, metrics_list=None, schedulers=None
                 device=torch.device("cuda:0"), ncritic=None, batch_size=128,
                 sample_size=8, epochs=5, checkpoints="./model/gan", retain_checkpoints=5,
                 recon="./images", test_noise=None, log_tensorboard=True, **kwargs):
        self.device = device
        self.model_names = []
        for key, val in models.items():
            self.model_names.append(key)
            if args in val:
                setattr(self, key, (val["name"](**val["args"])).to(self.device))
            else:
                setattr(self, key, (val["name"]()).to(self.device))
        self.optimizer_names = []
        for key, val in optimizers.items():
            self.optimizer_names.append(key)
            if args in val:
                setattr(self, key, val["name"](getattr(self, key.split("_", 1)[1]), **val["args"]))
            else:
                setattr(self, key, val["name"](getattr(self, key.split("_", 1)[1]))
        self.schedulers = []
        if schedulers is not None:
            for key, val in schedulers.items():
                if args in val:
                    self.schedulers.append(val["name"](getattr(self, key.split("_", 1)[1]), **val["args"]))
                else:
                    self.schedulers.append(val["name"](getattr(self, key.split("_", 1)[1])))
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
        self.test_noise = []
        for model in self.model_names:
            if isinstance(getattr(self, model), Generator):
                self.test_noise.append(torch.randn(self.sample_size, getattr(self, model).encoding_dims,
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
        self.labels_provided = False
        for key, val in kwargs.items():
            if key in self.__dict__():
                warn("Overiding the default value of {} from {} to {}".format(key, getattr(self, key), val))
            setattr(self, key, val)

    def save_model_extras(self, save_path):
        return {}

    def save_model(self, epoch):
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
        for save_items in self.model_names + self.optimizer_names:
            model.update({save_items: (getattr(self, save_items)).state_dict()})
        model.update(self.save_model_extras(save_path))
        torch.save(model, save_path)

    def load_model_extras(self, checkpoint):
        pass

    def load_model(self, load_path=""):
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
            # NOTE(avik-pal): Training might not occur in this case
            for load_items in self.model_names + self.optimizer_names:
                getattr(self, load_items).load_state_dict(checkpoint[load_items])
            self.load_model_extras(checkpoint)
        except:
            warn("Model could not be loaded from {}. Training from Scratch".format(load_path))

    # TODO(avik-pal): The _get_step will fail in a lot of cases
    def _get_step(self, update=True):
        if not update:
            return self.tensorboard_information["step"]
        if self.tensorboard_information["repeats"] < self.tensorboard_information["repeat_step"]:
            self.tensorboard_information["repeats"] += 1
            return self.tensorboard_information["step"]
        else:
            self.tensorboard_information["step"] += 1
            self.tensorboard_information["repeats"] = 1
            return self.tensorboard_information["step"]

    def sample_images(self, epoch):
        for model in self.model_names:
            if isinstance(getattr(self, model), Generator):
                save_path = "{}/epoch{}_{}.png".format(self.recon, epoch + 1,\
                            (datetime.now()).strftime("%H:%M:%S"))
                print("Generating and Saving Images to {}".format(save_path))
                generator = getattr(self, model)
                with torch.no_grad():
                    images = generator(self.test_noise.to(self.device))
                    img = torchvision.utils.make_grid(images)
                    torchvision.utils.save_image(img, save_path, nrow=self.nrow)
                    if self.log_tensorboard:
                        self.writer.add_image("Generated Samples", img, self._get_step(False))

    def train_logger(self, epoch, running_losses):
        print('Epoch {} Summary: '.format(epoch + 1))
        for name, val in running_losses.items():
            print('Mean {} : {}'.format(name, val))

    def tensorboard_log_losses(self):
        if self.log_tensorboard:
            running_generator_loss = self.loss_information["generator_losses"] /\
                self.loss_information["generator_iters"]
            running_discriminator_loss = self.loss_information["discriminator_losses"] /\
                self.loss_information["discriminator_iters"]
            self.writer.add_scalar("Running Discriminator Loss",
                                   running_discriminator_loss,
                                   self._get_step())
            self.writer.add_scalar("Running Generator Loss",
                                   running_generator_loss,
                                   self._get_step())
            self.writer.add_scalars("Running Losses",
                                   {"Running Discriminator Loss": running_discriminator_loss,
                                    "Running Generator Loss": running_generator_loss},
                                   self._get_step())

    def tensorboard_log_metrics(self):
        if self.tensorboard_log:
            for name, value in self.loss_logs.items():
                if type(value) is tuple:
                    self.writer.add_scalar('Losses/{}-Generator'.format(name), value[0], self._get_step(False))
                    self.writer.add_scalar('Losses/{}-Discriminator'.format(name), value[1], self._get_step(False))
                else:
                    self.writer.add_scalar('Losses/{}'.format(name), value, self._get_step(False))
            if self.metric_logs:
                for name, value in self.metric_logs.items():
                    # FIXME(Aniket1998): Metrics step should be number of epochs so far
                    self.writer.add_scalar("Metrics/{}".format(name),
                                           value, self._get_step(False))

    def set_arg_maps(self, mappings):
        if type(mappings) is list:
            for mapping in mappings:
                setattr(self, mapping[0], mapping[1])
        else:
            setattr(self, mappings[0], mappings[1])

    def _get_argument_maps(self, loss):
        sig = signature(loss.train_ops)
        args = list(sig.parameters.keys())
        for arg in args:
            if arg not in self.__dict__:
                raise Exception("Argument : %s needed for %s not present. If the value is stored with some other\
                                 name use the function `set_arg_maps`".format(arg, type(loss).__name__))
        return args

    def _store_loss_maps(self):
        self.loss_arg_maps = {}
        for name, loss in self.losses.items():
            self.loss_arg_maps[name] = self._get_argument_maps(loss)

    def train_stopper(self):
        if self.ncritic is None:
            return False
        else:
            return self.loss_information["discriminator_iters"] % self.ncritic != 0

    def train_iter_custom(self):
        pass

    # TODO(avik-pal): Clean up this function and avoid returning values
    def train_iter(self):
        self.train_iter_custom()
        ldis, lgen, dis_iter, gen_iter = 0.0, 0.0, 0, 0
        for name, loss in self.losses.items():
            if isinstance(loss, GeneratorLoss) and isinstance(loss, DiscriminatorLoss):
                cur_loss = loss.train_ops(*itemgetter(*self.loss_arg_maps[name])(self.__dict__))
                self.loss_logs[name].append(cur_loss)
                if type(cur_loss) is tuple:
                    lgen, ldis, gen_iter, dis_iter = lgen + cur_loss[0], ldis + cur_loss[1],\
                        gen_iter + 1, dis_iter + 1
            elif isinstance(loss, GeneratorLoss):
                if self.ncritic is None or\
                   self.loss_information["discriminator_iters"] % self.ncritic == 0:
                    cur_loss = loss.train_ops(*itemgetter(*self.loss_arg_maps[name])(self.__dict__))
                    self.loss_logs[name].append(cur_loss)
                    lgen, gen_iter = lgen + cur_loss, gen_iter + 1
            elif isinstance(loss, DiscriminatorLoss):
                cur_loss = loss.train_ops(*itemgetter(*self.loss_arg_maps[name])(self.__dict__))
                self.loss_logs[name].append(cur_loss)
                ldis, dis_iter = ldis + cur_loss, dis_iter + 1
        return lgen, ldis, gen_iter, dis_iter

    def log_metrics(self, epoch):
        if not self.metric_logs:
            warn('No evaluation metric logs present')
        else:
            for name, val in self.metric_logs.item():
                print('{} : {}'.format(name, val))
            self.tensorboard_log_metrics()

    def eval_ops(self, epoch, **kwargs):
        self.sample_images(epoch)
        if self.metrics is not None:
            for name, metric in self.metrics.items():
                if name + '_inputs' not in kwargs:
                    raise Exception("Inputs not provided for metric {}".format(name))
                else:
                    # NOTE(avik-pal): Needs to be changed if the user decides not to feed in generator and discriminator
                    self.metric_logs[name].append(metric.metric_ops(self.generator,
                                                                    self.discriminator, kwargs[name + '_inputs']))
                    self.log_metrics(self, epoch)

    def optim_ops(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def train(self, data_loader, **kwargs):
        for epoch in range(self.start_epoch, self.epochs):

            for model in self.model_names:
                getattr(self, model).train()

            for data in data_loader:
                if type(data) is tuple or type(data) is list:
                    self.real_inputs = data[0].to(self.device)
                    self.labels = data[1].to(self.device)
                else:
                    self.real_inputs = data

                lgen, ldis, gen_iter, dis_iter = self.train_iter()
                self.loss_information['generator_losses'] += lgen
                self.loss_information['discriminator_losses'] += ldis
                self.loss_information['generator_iters'] += gen_iter
                self.loss_information['discriminator_iters'] += dis_iter

                self.tensorboard_log_losses()

                if self.train_stopper():
                    break

            self.save_model(epoch)
            self.train_logger(epoch,
                              {'Generator Loss': self.loss_information['generator_losses'] /
                              self.loss_information['generator_iters'],
                              'Discriminator Loss': self.loss_information['discriminator_losses'] /
                              self.loss_information['discriminator_iters']})

            for model in self.model_names:
                getattr(self, model).eval()

            self.eval_ops(epoch, **kwargs)
            self.optim_ops()

        print("Training of the Model is Complete")

    def __call__(self, data_loader, **kwargs):
        self.writer = SummaryWriter()
        self._store_loss_maps()
        self.train(data_loader, **kwargs)
        self.writer.close()
