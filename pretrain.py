# code for pretrining the BLIP2 q-former model

import torch

from .blip_2_m import Blip2QFormerModel


def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    """Warmup the learning rate"""
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(model, data_loader, optimizer, epoch, device, config):
    # train
    model.train()

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    if config['laion_path']:
        data_loader.dataset.reload_laion(epoch)

    data_loader.sampler.set_epoch(epoch)

    for i, samples in enumerate(data_loader):

        if epoch == 0:
            warmup_lr_schedule(optimizer, i, config['warmup_steps'], config['warmup_lr'], config['init_lr'])

        optimizer.zero_grad()

        samples = samples.to(device, non_blocking=True)

        # ramp up alpha in the first 2 epochs
        alpha = config['alpha'] * min(1, (epoch * len(data_loader) + i) / (2 * len(data_loader)))

        loss, loss_ita, loss_itm, loss_lm = model(samples, alpha=alpha)

        loss.backward()
        optimizer.step()

        if i % print_freq == 0:
            print(header + ' Iter: [{} / {}] Loss: {:.4f} Loss_ITA: {:.4f} Loss_ITM: {:.4f} Loss_LM: {:.4f}'.format(
                i, len(data_loader), loss.item(), loss_ita.item(), loss_itm.item(), loss_lm.item()))
