import os
import gc
import time
import torch
import pickle
import argparse
import logging
import torchvision
import torch.utils.data
import matplotlib.pyplot as plt
from torchvision import transforms
from tensorboardX import SummaryWriter
from torch import nn
import numpy as np
import torchvision.utils as vutils
from PIL import Image
import json

from dataset import MovingObjectsDataset
from models import Generator, Discriminator
from losses import SSIM, L1_L2_Loss
from utils import init_log, add_file_handler, print_speed, Config, AverageMeter


def visualize(past_frames, true_future_frames, pred_future_frames,
              past_frames_test, true_future_frames_test, pred_future_frames_test,
              file_identifier, current_step, num_future_frame, writer):
    
    past_frames = torchvision.utils.make_grid(past_frames[0], num_future_frame)
    true_future_frames = torchvision.utils.make_grid(true_future_frames[0], num_future_frame)
    pred_future_frames = torchvision.utils.make_grid(pred_future_frames[0], num_future_frame)
    past_frames_test = torchvision.utils.make_grid(past_frames_test[0], num_future_frame)
    true_future_frames_test = torchvision.utils.make_grid(true_future_frames_test[0], num_future_frame)
    pred_future_frames_test = torchvision.utils.make_grid(pred_future_frames_test[0], num_future_frame)
    
    os.makedirs(f"../results/{file_identifier}/{current_step:06d}", exist_ok=True)

    plt.imsave(
        f"../results/{file_identifier}/{current_step:06d}/train_feed_seq.png",
        past_frames.cpu().permute(1, 2, 0).numpy()
    )
    plt.imsave(
        f"../results/{file_identifier}/{current_step:06d}/train_gt_seq.png",
        true_future_frames.cpu().permute(1, 2, 0).numpy(),
    )
    plt.imsave(
        f"../results/{file_identifier}/{current_step:06d}/train_pred_seq.png",
        pred_future_frames.detach().cpu().permute(1, 2, 0).numpy(),
    )
    plt.imsave(
        f"../results/{file_identifier}/{current_step:06d}/test_feed_seq.png",
        past_frames_test.cpu().permute(1, 2, 0).numpy(),
    )
    plt.imsave(
        f"../results/{file_identifier}/{current_step:06d}/test_gt_seq.png",
        true_future_frames_test.cpu().permute(1, 2, 0).numpy(),
    )
    plt.imsave(
        f"../results/{file_identifier}/{current_step:06d}/test_pred_seq.png",
        pred_future_frames_test.detach().cpu().permute(1, 2, 0).numpy(),
    )
    
    writer.add_image(f"train_feed_seq/{current_step:06d}", past_frames, current_step)
    writer.add_image(f"train_gt_seq/{current_step:06d}", true_future_frames, current_step)
    writer.add_image(f"train_pred_seq/{current_step:06d}", pred_future_frames, current_step)
    writer.add_image(f"test_feed_seq/{current_step:06d}", past_frames_test, current_step)
    writer.add_image(f"test_gt_seq/{current_step:06d}", true_future_frames_test, current_step)
    writer.add_image(f"test_pred_seq/{current_step:06d}", pred_future_frames_test, current_step)


def train(cfg):
    """
    Train loop
    :param cfg: Config file path
    """

    # Extract and set up training parameters
    cfg = Config(cfg)
    lr = cfg.train["lr"]
    epochs = cfg.train["epochs"]
    board_path = cfg.meta["board_path"]
    batch_size = cfg.train["batch_size"]
    num_future_frame = cfg.model["future_frames"]
    print_freq = cfg.train["print_frequency"]
    img_size = cfg.model["input_size"]
    train_data_dir = cfg.meta["data_path_local_train"]
    test_data_dir = cfg.meta["data_path_local_test"]
    file_identifier = f'{batch_size}_{epochs}_{img_size[0]}{img_size[1]}_hpc'

    avg = AverageMeter()

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataloaders for train and test dataset
    train_dataset = MovingObjectsDataset(
        root_dir=f'{train_data_dir}',
        transform=transforms.Compose(
                [
                    transforms.Resize((img_size[0], img_size[1])),
                    transforms.ToTensor(),
                ]
            )
        )
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1)


    test_dataset = MovingObjectsDataset(
        root_dir=f'{test_data_dir}',
        transform=transforms.Compose(
                [
                    transforms.Resize((img_size[0], img_size[1])),
                    transforms.ToTensor(),
                ]
            )
        )
    
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1)

    test_iter = iter(test_loader)

    os.makedirs(os.path.join(os.getcwd(), f"logs/{file_identifier}"), exist_ok=True)
    global_logger = init_log("global", level=logging.INFO)
    add_file_handler("global", os.path.join(os.getcwd(), f"logs/{file_identifier}", "train.log"), level=logging.DEBUG)

    global_logger.debug("==>>> Total training batches: {}".format(len(train_loader)))
    global_logger.debug("==>>> Total testing batches: {}".format(len(test_loader)))

    writer = SummaryWriter(os.path.join(".", board_path))

    # Create Generator and Discriminator models
    generator = Generator(cfg=cfg.model, device=device)
    generator.to(device)

    discriminator = Discriminator(cfg=cfg.model)
    discriminator.to(device)

    # Define optimizers and LR Schedulers
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)

    generator_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=generator_optimizer, milestones=[10, 20, 30, 40], gamma=0.5
    )
    discriminator_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=discriminator_optimizer, milestones=[10, 20, 30, 40], gamma=0.5
    )

    # Distribute training across multiple GPUs
    generator = nn.DataParallel(generator)
    discriminator = nn.DataParallel(discriminator)

    # Begin training (restart from checkpoint if possible)
    start_epoch = 0
    if os.path.isfile(f"../model/model_{file_identifier}.pt"):
        print("Restarting training...")
        checkpoint = torch.load(f"../model/model_{file_identifier}.pt")
        generator.load_state_dict(checkpoint["generator_state_dict"])
        generator_optimizer.load_state_dict(checkpoint["generator_optimizer_state_dict"])
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]

    # Define losses
    ssim_loss = SSIM(window_size=11, size_average=True).to(device)
    l1_l2_loss = L1_L2_Loss().to(device)
    adversarial_loss = torch.nn.BCELoss().to(device)

    # Value trackers
    train_loss_list = []
    test_loss_list = []
    train_metric_list = []
    test_metric_list = []

    # Train loop
    for epoch in range(start_epoch, epochs):
        for step, [past_frames, true_future_frames] in enumerate(train_loader):
            start_time = time.time()

            generator.train()
            discriminator.train()

            if epoch == 0 and step == 0:
                global_logger.debug("Input:  {}".format(past_frames.shape))
                global_logger.debug("--- Sample")
                global_logger.debug("Target: {}".format(true_future_frames.shape))

            past_frames, true_future_frames = past_frames.to(device), true_future_frames.to(device)
            pred_future_frames = generator(past_frames, future=num_future_frame)

            # Train discriminator to classify real and predicted frames with label smoothing
            discriminator.zero_grad()
            seq_target_frames = true_future_frames.squeeze().view(-1, *true_future_frames.shape[3:])
            label = torch.empty(seq_target_frames.size(0), device=device).uniform_(0.9, 1)
            output = discriminator(seq_target_frames).view(-1)
            discriminator_loss_real = adversarial_loss(output, label)

            predicted_future_frames_individual = pred_future_frames.squeeze().view(-1, *true_future_frames.shape[3:])
            label = torch.empty(predicted_future_frames_individual.size(0), device=device).uniform_(0, 0.1)
            output = discriminator(predicted_future_frames_individual).view(-1)
            discriminator_loss_fake = adversarial_loss(output, label)

            discriminator_loss = discriminator_loss_real + discriminator_loss_fake
            discriminator_loss.backward(retain_graph=True)
            discriminator_optimizer.step()

            # Train generator with adversarial loss with label smoothing
            generator.zero_grad()
            label = torch.empty(predicted_future_frames_individual.size(0), device=device).uniform_(0.9, 1)
            output = discriminator(predicted_future_frames_individual).view(-1)

            # Weighted loss for generator model with emphasis on image quality
            generator_loss = adversarial_loss(output, label) + 4 * l1_l2_loss(
                pred_future_frames[:, -num_future_frame:, :, :, :], true_future_frames[:, -num_future_frame:, :, :, :]
            )

            generator_loss.backward(retain_graph=True)
            generator_optimizer.step()

            # Evaluate and test the model
            with torch.no_grad():
                train_metric = ssim_loss(
                    pred_future_frames[:, -num_future_frame:, :, :, :],
                    true_future_frames[:, -num_future_frame:, :, :, :],
                )

                try:
                    past_frames_test, true_future_frames_test = next(test_iter)
                except StopIteration:
                    test_iter = iter(test_loader)
                    past_frames_test, true_future_frames_test = next(test_iter)

                past_frames_test = past_frames_test.to(device)
                true_future_frames_test = true_future_frames_test.to(device)
                pred_future_frames_test = generator(past_frames_test, future=num_future_frame)

                test_loss = l1_l2_loss(
                    pred_future_frames_test[:, -num_future_frame:, :, :, :],
                    true_future_frames_test[:, -num_future_frame:, :, :, :],
                )
                test_metric = ssim_loss(
                    pred_future_frames_test[:, -num_future_frame:, :, :, :],
                    true_future_frames_test[:, -num_future_frame:, :, :, :],
                )

            # Log values and store the model outputs on main worker
            step_time = time.time() - start_time

            train_loss_list.append(generator_loss.item())
            test_loss_list.append(test_loss.item())
            train_metric_list.append(train_metric.item())
            test_metric_list.append(test_metric.item())

            

            if (step + 1) % print_freq == 0:
                current_step = epoch * len(train_loader) + step + 1
                
                visualize(past_frames, true_future_frames, pred_future_frames,
                          past_frames_test, true_future_frames_test, pred_future_frames_test,
                          file_identifier, current_step, num_future_frame, writer)

                torch.save(
                    {
                        "epoch": epoch,
                        "generator_state_dict": generator.state_dict(),
                        "generator_optimizer_state_dict": generator_optimizer.state_dict(),
                        "discriminator_state_dict": discriminator.state_dict(),
                        "discriminator_optimizer_state_dict": discriminator_optimizer.state_dict(),
                    },
                    f"../model/model_{file_identifier}.pt",
                )

            writer.add_scalars(
                f"loss/{file_identifier}/merge",
                {
                    "generator_loss": generator_loss.item(),
                    "discriminator_loss": discriminator_loss.item(),
                    "test_loss": test_loss.item(),
                    "train_metric": train_metric.item(),
                    "test_metric": test_metric.item(),
                },
                epoch * len(train_loader) + step + 1,
            )

            avg.update(
                step_time=step_time,
                generator_loss=generator_loss.item(),
                discriminator_loss=discriminator_loss.item(),
                test_loss=test_loss.item(),
                train_metric=train_metric.item(),
            )

            if (step + 1) % print_freq == 0:
                global_logger.info(
                    "Epoch: [{0}][{1}/{2}] {Step_Time:s}\t{Gen_loss:s}\t{Disc_loss:s}\t{Test_loss:s}\t{Train_metric:s}".format(
                        epoch + 1,
                        (step + 1) % len(train_loader),
                        len(train_loader),
                        Step_Time=avg.step_time,
                        Gen_loss=avg.generator_loss,
                        Disc_loss=avg.discriminator_loss,
                        Test_loss=avg.test_loss,
                        Train_metric=avg.train_metric,
                    )
                )
                print_speed(epoch * len(train_loader) + step + 1, avg.step_time.avg, epochs * len(train_loader))

        generator_scheduler.step()
        discriminator_scheduler.step()
        gc.collect()

    with open(f"../results/{file_identifier}/train_loss_list.pkl", "wb") as f:
        pickle.dump(train_loss_list, f)
    with open(f"../results/{file_identifier}/test_loss_list.pkl", "wb") as f:
        pickle.dump(test_loss_list, f)

    with open(f"../results/{file_identifier}/train_metric_list.pkl", "wb") as f:
        pickle.dump(train_metric_list, f)
    with open(f"../results/{file_identifier}/test_metric_list.pkl", "wb") as f:
        pickle.dump(test_metric_list, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--cfg",
        default=os.path.join(os.getcwd(), "config.json"),
        type=str,
        required=False,
        help="Training config file path",
    )
    args = parser.parse_args()
    train(args.cfg)
    
