import torch
import torchvision
from tensorboardX import SummaryWriter
from ..models.model import Generator, Discriminator

__all__ = ['Visualize', 'LossVisualize', 'MetricVisualize',
           'GradientVisualize', 'ImageVisualize']

class Visualize(object):
    def __init__(self, visualize_list, tensorboard=True, log_dir=None, writer=None):
        self.writer = SummaryWriter(log_dir) if writer is None else writer
        self.logs = {}
        for item in visualize_list:
            name = type(item).__name__
            self.logs[name] = []
        self.tensorboard = tensorboard
        self.tensorboard_step = 1

    def tensorboard_step_update(self):
        self.tensorboard_step += 1

    def log_tensorboard(self):
        pass

    def log_items(self):
        pass

    def __call__(self, *args, lock_items=False, lock_tensorboard=False, **kwargs):
        if not lock_items:
            self.log_items(*args, **kwargs)
        if self.tensorboard and not lock_tensorboard:
            self.log_tensorboard(*args, **kwargs)
        self.tensorboard_step_update()

class LossVisualize(Visualize):
    def __init__(self, *args, **kwargs):
        super(Visualize, self).__init__(*args, **kwargs)

    def log_tensorboard(self, running_losses):
        self.writer.add_scalar("Running Discriminator Loss",
                               running_losses["Running Discriminator Loss"],
                               self.tensorboard_step)
        self.writer.add_scalar("Running Generator Loss",
                               running_losses["Running Generator Loss"],
                               self.tensorboard_step)
        self.writer.add_scalars("Running Losses",
                                running_losses,
                                self.tensorboard_step)
        for name, value in self.logs.items():
            val = value[-1]
            if type(val) is tuple:
                self.writer.add_scalar('Losses/{}-Generator'.format(name), val[0], self.tensorboard_step)
                self.writer.add_scalar('Losses/{}-Discriminator'.format(name), val[1], self.tensorboard_step)
            else:
                self.writer.add_scalar('Losses/{}'.format(name), val, self._get_step(False))

    def log_items(self, running_losses):
        for name, val in running_losses.items():
            print('Mean {} : {}'.format(name, val))

    def __call__(self, trainer, **kwargs):
        running_generator_loss = trainer.loss_information["generator_losses"] /\
            trainer.loss_information["generator_iters"]
        running_discriminator_loss = trainer.loss_information["discriminator_losses"] /\
            trainer.loss_information["discriminator_iters"]
        running_losses = {"Running Discriminator Loss": running_discriminator_loss,
                          "Running Generator Loss": running_generator_loss}
        super(Visualize, self).__call__(running_losses, **kwargs)

class MetricVisualize(Visualize):
    def __init__(self, *args, **kwargs):
        super(Visualize, self).__init__(*args, **kwargs)

    def log_tensorboard(self):
        for name, value in self.logs.items():
            self.writer.add_scalar("Metrics/{}".format(name), value[-1], self.tensorboard_step)

    def log_items(self):
        for name, val in self.logs.items():
            print('{} : {}'.format(name, val[-1]))

class GradientVisualize(Visualize):
    def __init__(self, *args, **kwargs):
        super(Visualize, self).__init__(*args, **kwargs)

    def log_tensorboard(self, name, gradsum):
        self.writer.add_scalar('Gradients/{}'.format(name), gradsum, self.tensorboard_step)

    def log_items(self, name, gradsum):
        print('{} Gradients : {}'.format(name, gradsum))

    def __call__(self, trainer, **kwargs):
        for name in trainer.model_names:
            model = getattr(trainer, name)
            gradsum = 0.0
            for p in model.parameters():
                gradsum += p.norm(2).item()
            super(Visualize, self).__call__(name, gradsum, **kwargs)

class ImageVisualize(Visualize):
    def __init__(self, trainer, tensorboard=True, log_dir=None, writer=None,
                 test_noise=None, nrow=8):
        super(Visualize, self).__init__([], writer=True) # Samll hack
        self.writer = SummaryWriter(log_dir) if writer is None else writer
        self.test_noise = []
        for model in trainer.model_names:
            if isinstance(getattr(trainer, model), Generator):
                self.test_noise.append(getattr(trainer, model).sampler(trainer.sample_size, trainer.device)
                                       if test_noise is None else test_noise)
        self.tensorboard = tensorboard
        self.tensorboard_step = 1
        self.nrow = nrow

    def log_tensorboard(self, trainer, image, model):
        self.writer.add_image("Generated Samples/{}".format(model), image, self.tensorboard_step)

    def log_items(self, trainer, image, model):
        save_path = "{}/epoch{}_{}.png".format(trainer.recon, self.tensorboard_step, model)
        print("Generating and Saving Images to {}".format(save_path))
        torchvision.utils.save_image(image, save_path, nrow=self.nrow)

    def __call__(self, trainer, **kwargs):
        pos = 0
        for model in trainer.model_names:
            if isinstance(getattr(trainer, model), Generator):
                generator = getattr(trainer, model)
                with torch.no_grad():
                    image = generator(*self.test_noise[pos])
                    image = torchvision.utils.make_grid(image)
                    super(Visualize, self).__call__(trainer, image, model, **kwargs)
                pos = pos + 1
