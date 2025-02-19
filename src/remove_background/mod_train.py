import os
import time
from math import ceil
import itertools

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler

from .model.mod_unet_pp import build_model
from .datasets import ModMagazineCropDataset, mod_mc_collate_fn
from .mod_transforms import (
    build_scanned_transform,
)
from ..utils.arg_parser import get_parser
from ..utils.misc import move_to_device
from .loss import ModComboLoss
from .metrics import IOUMetric

# to download model's weights, execute the following command:
# scp <username>@<ip>:/home/ubuntu/projects/MagazineCrop/src/remove_background/checkpoints/<model_name> ./src/remove_background/checkpoints/


def poly_lr(
    epoch,
    base_lr: float = 1e-3,
    min_lr: float = 5e-5,
    max_epochs: int = 20,
    power: float = 0.9,
):
    factor = (1 - epoch / max_epochs) ** power
    return max(min_lr / base_lr, factor)


def train(
    args,
    model: nn.Module,
    optimizer,
    loss_fn: nn.Module,
    data_loader: dict[str, DataLoader],
    epochs: int,
    accumulation_steps: int = 8,
    scheduler: torch.optim.lr_scheduler = None,
    metrics_fn: dict[str, nn.Module] = {},
):
    print("Training model...")

    start_epoch = 0
    iou_record = float("-inf")
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
        iou_record = checkpoint["iou"]
        start_epoch = checkpoint["epoch"] + 1

    model = model.to(args.device)
    path_to_save = os.path.join(args.checkpoint_dir, args.save_as)

    best_loss = float("inf")
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        model.train()

        optimizer.zero_grad()
        running_loss = 0.0
        metrics = dict.fromkeys(metrics_fn.keys(), 0.0)
        for ind, data in enumerate(data_loader["train"], start=1):
            inputs, targets, weights = [
                move_to_device(data=x, device=args.device) for x in data
            ]

            # outputs = {'logits': logits, 'coords': coords}
            outputs = model(
                src=inputs["images"],
                edge_length=inputs["length"],
                edge_theta=inputs["theta"],
            )

            loss, *_ = loss_fn(outputs=outputs, targets=targets, weights=weights)
            running_loss += loss.item()

            loss = loss / accumulation_steps
            loss.backward()

            if ind % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            for metric_name, metric_fn in metrics_fn.items():
                metrics[metric_name] += metric_fn(
                    outputs=outputs["coords"].detach(),
                    targets=targets["corner_coordinates"].detach(),
                ).item()

        remainder = ind % accumulation_steps
        if remainder > 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        avg_loss = running_loss / ind
        iou_metrics = metrics.get("iou", 0.0) / ind
        print(
            f"Epoch {epoch+1:>2}:\n\t{'Train Loss':<20}: {avg_loss:.6f}\n\t{'Train IoU':<20}: {iou_metrics:.6f}"
        )

        running_vloss = 0.0
        vmetrics = dict.fromkeys(metrics_fn.keys(), 0.0)

        model.eval()
        with torch.no_grad():
            for ind, data in enumerate(data_loader["valid"], start=1):
                inputs, targets, weights = [
                    move_to_device(data=x, device=args.device) for x in data
                ]

                # outputs = {'logits': logits, 'coords': coords}
                outputs = model(
                    src=inputs["images"],
                    edge_length=inputs["length"],
                    edge_theta=inputs["theta"],
                )

                loss, *_ = loss_fn(outputs=outputs, targets=targets, weights=weights)

                for metric_name, metric_fn in metrics_fn.items():
                    vmetrics[metric_name] += metric_fn(
                        outputs=outputs["coords"],
                        targets=targets["corner_coordinates"],
                    ).item()
                running_vloss += loss.item()

        avg_vloss = running_vloss / ind
        combined_loss = avg_loss * 0.2 + avg_vloss * 0.8

        iou_vmetrics = vmetrics.get("iou", 0.0) / ind
        combined_iou_metrics = iou_metrics * 0.2 + iou_vmetrics * 0.8

        output_valid_message = ""
        output_valid_message += f"\t{'Valid Loss':<20}: {avg_vloss:.6f}\n"
        output_valid_message += f"\t{'Valid IoU':<20}: {iou_vmetrics:.6f}\n"
        output_valid_message += "\n"

        if combined_iou_metrics > iou_record:
            iou_record = combined_iou_metrics
            if not args.no_save:
                if not os.path.isdir(path_to_save):
                    os.mkdir(path_to_save)
                torch.save(model, os.path.join(path_to_save, f"{args.save_as}.pth"))
                states = {
                    "epoch": epoch,
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "iou": combined_iou_metrics,
                }
                torch.save(states, os.path.join(path_to_save, "checkpoint.pth"))
            output_valid_message += "\tNew Record, Saved!"
        print(output_valid_message)
        print(f"\t{'Best IoU':<20}: {iou_record:.6f}")

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
    metrics_fn: dict[str, nn.Module] = {},
):
    print("Training model...")

    start_epoch = 0
    iou_record = float("-inf")
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
        iou_record = checkpoint["iou"]
        start_epoch = checkpoint["epoch"] + 1

    scaler = GradScaler()

    model = model.to(args.device)
    path_to_save = os.path.join(args.checkpoint_dir, args.save_as)

    best_loss = float("inf")
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        model.train()

        optimizer.zero_grad()
        running_loss = 0.0
        metrics = dict.fromkeys(metrics_fn.keys(), 0.0)
        for ind, data in enumerate(data_loader["train"], start=1):
            inputs, targets, weights = [
                move_to_device(data=x, device=args.device) for x in data
            ]

            # outputs = {'logits': logits, 'coords': coords}
            with torch.autocast(device_type=args.device, dtype=torch.float16):
                outputs = model(
                    src=inputs["images"],
                    edge_length=inputs["length"],
                    edge_theta=inputs["theta"],
                )

                loss, *_ = loss_fn(outputs=outputs, targets=targets, weights=weights)
                running_loss += loss.item()
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if ind % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            for metric_name, metric_fn in metrics_fn.items():
                metrics[metric_name] += metric_fn(
                    outputs=outputs["coords"].detach(),
                    targets=targets["corner_coordinates"].detach(),
                ).item()

        remainder = ind % accumulation_steps
        if remainder > 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        avg_loss = running_loss / ind
        iou_metrics = metrics.get("iou", 0.0) / ind
        print(
            f"Epoch {epoch+1:>2}:\n\t{'Train Loss':<20}: {avg_loss:.6f}\n\t{'Train IoU':<20}: {iou_metrics:.6f}"
        )

        running_vloss = 0.0
        vmetrics = dict.fromkeys(metrics_fn.keys(), 0.0)

        model.eval()
        with torch.no_grad():
            for ind, data in enumerate(data_loader["valid"], start=1):
                inputs, targets, weights = [
                    move_to_device(data=x, device=args.device) for x in data
                ]

                with torch.autocast(device_type=args.device, dtype=torch.float16):
                    # outputs = {'logits': logits, 'coords': coords}
                    outputs = model(
                        src=inputs["images"],
                        edge_length=inputs["length"],
                        edge_theta=inputs["theta"],
                    )

                    loss, *_ = loss_fn(
                        outputs=outputs, targets=targets, weights=weights
                    )

                for metric_name, metric_fn in metrics_fn.items():
                    vmetrics[metric_name] += metric_fn(
                        outputs=outputs["coords"],
                        targets=targets["corner_coordinates"],
                    ).item()
                running_vloss += loss.item()

        avg_vloss = running_vloss / ind
        combined_loss = avg_loss * 0.2 + avg_vloss * 0.8

        iou_vmetrics = vmetrics.get("iou", 0.0) / ind
        combined_iou_metrics = iou_metrics * 0.2 + iou_vmetrics * 0.8

        output_valid_message = ""
        output_valid_message += f"\t{'Valid Loss':<20}: {avg_vloss:.6f}\n"
        output_valid_message += f"\t{'Valid IoU':<20}: {iou_vmetrics:.6f}\n"
        output_valid_message += "\n"

        if combined_iou_metrics > iou_record:
            iou_record = combined_iou_metrics
            if not args.no_save:
                if not os.path.isdir(path_to_save):
                    os.mkdir(path_to_save)
                torch.save(model, os.path.join(path_to_save, f"{args.save_as}.pth"))
                states = {
                    "epoch": epoch,
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "iou": combined_iou_metrics,
                }
                torch.save(states, os.path.join(path_to_save, "checkpoint.pth"))

            output_valid_message += "\tNew Record, Saved!"
        print(output_valid_message)
        print(f"\t{'Best IoU':<20}: {iou_record:.6f}")

        epoch_end = time.time()
        min_t = (epoch_end - epoch_start) // 60
        sec_t = int((epoch_end - epoch_start) % 60)
        print(f"\t{'Epoch Time':<20}: {min_t} min(s) {sec_t} sec(s)\n")


if __name__ == "__main__":
    parser = get_parser("dev")
    args = parser.parse_args()

    src_shape = (args.edge_size, args.edge_size)

    train_dataset = ModMagazineCropDataset(
        split="train",
        transforms=build_scanned_transform(split="train", size=src_shape),
        augment_factor=args.augment_factor,
        edge_size=args.edge_size,
    )
    valid_dataset = ModMagazineCropDataset(
        split="valid",
        transforms=build_scanned_transform(split="valid", size=src_shape),
        augment_factor=2,
        edge_size=args.edge_size,
    )
    dataloader = {
        "train": DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.dataloader_workers,
            collate_fn=mod_mc_collate_fn,
        ),
        "valid": DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.dataloader_workers,
            collate_fn=mod_mc_collate_fn,
        ),
    }

    model = build_model(
        path_to_ckpt=args.checkpoint_dir,
        src_shape=src_shape,
    )
    optimizer = torch.optim.AdamW(
        [
            {
                "params": itertools.chain(
                    model._backbone.parameters(), model._to_logits.parameters()
                ),
                "lr": args.lr_backbone,
            },
            {"params": model._line_approx_block.parameters()},
        ],
        lr=args.learning_rate,
        eps=1e-5 if args.mixed_precision else 1e-8,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.lr_backbone * 10, args.learning_rate * 10],
        total_steps=ceil(len(dataloader["train"]) / args.accumulation_steps)
        * args.epochs,
        pct_start=0.3,
    )

    loss_fn = ModComboLoss(number_of_classes=1, factor=0.3)
    iou_metric = IOUMetric(
        height=src_shape[0], width=src_shape[1], reduction="mean"
    ).to(args.device)

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
            metrics_fn={"iou": iou_metric},
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
            metrics_fn={"iou": iou_metric},
        )
