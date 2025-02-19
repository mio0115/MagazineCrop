import os
import time
from math import ceil
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler

# from .model.model_unet import build_iterative_model
from .model.model_unet3p import build_model
from .datasets import MyVOCSegmentation, MagazineCropDataset, mc_collate_fn
from .transforms import (
    build_scanned_transform,
)
from ..utils.misc import move_to_device
from ..utils.arg_parser import get_parser
from .loss import ComboLoss
from .metrics import IOUMetric

# to download model's weights, execute the following command:
# scp <username>@<ip>:/home/ubuntu/projects/MagazineCrop/src/remove_background/checkpoints/<model_name> ./src/remove_background/checkpoints/


def train(
    args,
    model: nn.Module,
    optimizer,
    loss_fn: nn.Module,
    data_loader: dict[str, DataLoader],
    epochs: int,
    accumulation_steps: int = 8,
    scheduler: torch.optim.lr_scheduler = None,
):
    print("Training model...")

    start_epoch = 0
    best_loss = float("inf")
    if args.resume:
        model = torch.load(
            os.path.join(
                args.checkpoint_dir, args.resume_from, f"{args.resume_from}.pth"
            ),
            weights_only=False,
        )
        checkpoint = torch.load(
            os.path.join(args.checkpoint_dir, args.resume_from, "checkpoint.pth")
        )
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        best_loss = checkpoint["loss"]
        start_epoch = checkpoint["epoch"] + 1

    model = model.to(args.device)
    path_to_save = os.path.join(args.checkpoint_dir, args.save_as)

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        model.train()

        optimizer.zero_grad()
        running_loss = 0.0
        for ind, data in enumerate(data_loader["train"], start=1):
            inputs, targets, weights = [
                move_to_device(data=x, device=args.device) for x in data
            ]

            # outputs = {'logits': logits,}
            outputs = model(
                src=inputs["images"],
            )

            loss = loss_fn(
                logits=outputs["logits"],
                targets=targets["labels"],
                weights=weights,
            )
            running_loss += loss.item()
            loss = loss / accumulation_steps

            loss.backward()

            if ind % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

        remainder = ind % accumulation_steps
        if remainder > 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        avg_loss = running_loss / ind
        print(f"Epoch {epoch+1:>2}:\n\t{'Train Loss':<20}: {avg_loss:.6f}")

        running_vloss = 0.0

        model.eval()
        with torch.no_grad():
            for ind, data in enumerate(data_loader["valid"], start=1):
                inputs, targets, weights = [
                    move_to_device(data=x, device=args.device) for x in data
                ]

                # outputs = {'logits': logits}
                outputs = model(
                    src=inputs["images"],
                )

                loss = loss_fn(
                    logits=outputs["logits"],
                    targets=targets["labels"],
                    weights=weights,
                )

                running_vloss += loss.item()

        avg_vloss = running_vloss / ind

        output_valid_message = ""
        output_valid_message += f"\t{'Valid Loss':<20}: {avg_vloss:.6f}\n"

        if avg_vloss < best_loss:
            best_loss = avg_vloss
            if not args.no_save:
                if not os.path.isdir(path_to_save):
                    os.mkdir(path_to_save)
                torch.save(model, os.path.join(path_to_save, f"{args.save_as}.pth"))
                states = {
                    "epoch": epoch,
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "loss": best_loss,
                }
                torch.save(states, os.path.join(path_to_save, "checkpoint.pth"))

            output_valid_message += "\tNew Record, Saved!"
        print(output_valid_message)
        print(f"\t{'Best Loss':<20}: {best_loss:.6f}")

        epoch_end = time.time()
        min_t = (epoch_end - epoch_start) // 60
        sec_t = int((epoch_end - epoch_start) % 60)
        print(f"\t{'Epoch Time':<20}: {min_t} min(s) {sec_t} sec(s)\n")


def mixed_precision_train(
    args,
    model: nn.Module,
    optimizer,
    loss_fn: nn.Module,
    data_loader: dict[str, DataLoader],
    epochs: int,
    accumulation_steps: int = 8,
    scheduler: torch.optim.lr_scheduler = None,
    metrics: Optional[dict[str, nn.Module]] = None,
):
    print("Training model...")

    start_epoch = 0
    best_iou = 0
    scaler = GradScaler()
    if args.resume:
        model = torch.load(
            os.path.join(
                args.checkpoint_dir, args.resume_from, f"{args.resume_from}.pth"
            ),
            weights_only=False,
        )
        checkpoint = torch.load(
            os.path.join(args.checkpoint_dir, args.resume_from, "checkpoint.pth")
        )
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        best_iou = checkpoint["iou"]
        start_epoch = checkpoint["epoch"] + 1

    model = model.to(args.device)
    path_to_save = os.path.join(args.checkpoint_dir, args.save_as)

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        model.train()

        optimizer.zero_grad()
        running_loss = 0.0
        for ind, data in enumerate(data_loader["train"], start=1):
            inputs, targets, weights = [
                move_to_device(data=x, device=args.device) for x in data
            ]

            # outputs = {'logits': logits,}
            with torch.autocast(device_type=args.device, dtype=torch.float16):
                outputs = model(
                    src=inputs["images"],
                )

                loss = loss_fn(
                    logits=outputs["logits"],
                    targets=targets["labels"],
                    weights=weights,
                )

            if not torch.isnan(loss).item():
                running_loss += loss.item()
            loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if ind % accumulation_steps == 0:
                # for name, param in model.named_parameters():
                #     if param.grad is not None:
                #         print(
                #             f"{name:<40}: Mean Grad = {param.grad.abs().mean():.6f}, Max Grad = {param.grad.abs().max():.6f}"
                #         )

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                scheduler.step()

        remainder = ind % accumulation_steps
        if remainder > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        avg_loss = running_loss / ind
        print(f"Epoch {epoch+1:>2}:\n\t{'Train Loss':<20}: {avg_loss:.6f}")

        running_vloss = 0.0
        running_viou = 0.0

        model.eval()
        with torch.no_grad():
            for ind, data in enumerate(data_loader["valid"], start=1):
                inputs, targets, weights = [
                    move_to_device(data=x, device=args.device) for x in data
                ]

                with torch.autocast(device_type=args.device, dtype=torch.float16):
                    # outputs = {'logits': logits}
                    outputs = model(
                        src=inputs["images"],
                    )

                    loss = loss_fn(
                        logits=outputs["logits"],
                        targets=targets["labels"],
                        weights=weights,
                    )

                    viou = metrics["iou"](outputs["logits"], targets["labels"])
                    running_viou += viou.item()

                running_vloss += loss.item()

        avg_vloss = running_vloss / ind
        avg_viou = running_viou / ind

        output_valid_message = ""
        output_valid_message += f"\t{'Valid Loss':<20}: {avg_vloss:.6f}\n"

        if avg_viou > best_iou:
            best_iou = avg_viou
            if not args.no_save:
                if not os.path.isdir(path_to_save):
                    os.mkdir(path_to_save)
                torch.save(model, os.path.join(path_to_save, f"{args.save_as}.pth"))
                states = {
                    "epoch": epoch,
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "iou": best_iou,
                }
                torch.save(states, os.path.join(path_to_save, "checkpoint.pth"))

            output_valid_message += "\tNew Record, Saved!"
        print(output_valid_message)
        print(f"\t{'Best IoU':<20}: {best_iou:.6f}")

        epoch_end = time.time()
        min_t = (epoch_end - epoch_start) // 60
        sec_t = int((epoch_end - epoch_start) % 60)
        print(f"\t{'Epoch Time':<20}: {min_t} min(s) {sec_t} sec(s)\n")


if __name__ == "__main__":
    parser = get_parser("dev")
    args = parser.parse_args()

    reshape_size = args.edge_size

    train_dataset = MagazineCropDataset(
        split="train",
        transforms=build_scanned_transform(split="train", reshape_size=reshape_size),
        augment_factor=args.augment_factor,
    )
    valid_dataset = MagazineCropDataset(
        split="valid",
        transforms=build_scanned_transform(split="valid", reshape_size=reshape_size),
        augment_factor=1,
    )
    dataloader = {
        "train": DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.dataloader_workers,
            collate_fn=mc_collate_fn,
        ),
        "valid": DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.dataloader_workers,
            collate_fn=mc_collate_fn,
        ),
    }

    model = build_model(in_channels=3, out_channels=1)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoders.parameters(), "lr": args.lr_backbone},
            {
                "params": [
                    param
                    for key, param in model.named_parameters()
                    if "encoders" not in key
                ],
                "lr": args.learning_rate,
            },
        ],
        eps=1e-5 if args.mixed_precision else 1e-8,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.lr_backbone * 10, args.learning_rate * 10],
        total_steps=ceil(len(dataloader["train"]) / args.accumulation_steps)
        * args.epochs,
        pct_start=0.3,
    )
    loss_fn = ComboLoss(number_of_classes=1)
    iou_metric = IOUMetric(threshold=0.5, reduction="mean")

    if args.mixed_precision:
        mixed_precision_train(
            args,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            data_loader=dataloader,
            epochs=args.epochs,
            accumulation_steps=args.accumulation_steps,
            scheduler=scheduler,
            metrics={"iou": iou_metric},
        )
    else:
        train(
            args,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            data_loader=dataloader,
            epochs=args.epochs,
            accumulation_steps=args.accumulation_steps,
            scheduler=scheduler,
        )
