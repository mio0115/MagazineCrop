import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from .model.mod_unet_pp import build_model
from .datasets import ModMagazineCropDataset, mod_mc_collate_fn
from .mod_transforms import (
    build_scanned_transform,
)
from ..utils.arg_parser import get_parser
from .loss import ModComboLoss
from .metrics import IOUMetric

# to download model's weights, execute the following command:
# scp <username>@<ip>:/home/ubuntu/projects/MagazineCrop/src/remove_background/checkpoints/<model_name> ./src/remove_background/checkpoints/


def move_to_device(
    data: dict[str, torch.Tensor] | torch.Tensor, device: str | torch.device
) -> dict[str, torch.Tensor]:
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: v.to(device) for k, v in data.items()}
    else:
        raise ValueError("Data should be either a dict or a tensor")


def train(
    args,
    model: nn.Module,
    optimizer,
    loss_fn: nn.Module,
    data_loader: dict[str, DataLoader],
    epochs: int,
    valid: bool = True,
    metrics_fn: dict[str, nn.Module] = {},
):
    print("Training model...")
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10])
    model = model.to(args.device)
    path_to_save = os.path.join(args.checkpoint_dir, args.save_as)

    best_loss = float("inf")
    iou_record = float("-inf")
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()

        running_loss = 0.0
        metrics = dict.fromkeys(metrics_fn.keys(), 0.0)
        for ind, data in enumerate(data_loader["train"]):
            inputs, targets, weights = [
                move_to_device(data=x, device=args.device) for x in data
            ]

            optimizer.zero_grad()
            # outputs = {'logits': logits, 'coords': coords}
            outputs = model(
                src=inputs["images"],
                edge_length=inputs["length"],
                edge_theta=inputs["theta"],
            )

            loss, *_ = loss_fn(outputs=outputs, targets=targets, weights=weights)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            for metric_name, metric_fn in metrics_fn.items():
                metrics[metric_name] += metric_fn(
                    outputs=outputs["coords"].detach(),
                    targets=targets["corner_coordinates"].detach(),
                ).item()

        scheduler.step()
        avg_loss = running_loss / (ind + 1)
        print(f"Epoch {epoch+1:>2}:\n\t{'Train Loss':<20}: {avg_loss:.6f}")

        iou_metrics = metrics.get("iou", 0.0) / (ind + 1)

        if valid:
            running_vloss = 0.0
            vmetrics = dict.fromkeys(metrics_fn.keys(), 0.0)

            model.eval()
            with torch.no_grad():
                for ind, data in enumerate(data_loader["valid"]):
                    inputs, targets, weights = [
                        move_to_device(data=x, device=args.device) for x in data
                    ]

                    optimizer.zero_grad()
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

            avg_vloss = running_vloss / (ind + 1)
            combined_loss = avg_loss * 0.2 + avg_vloss * 0.8

            iou_vmetrics = vmetrics.get("iou", 0.0) / (ind + 1)
            combined_iou_metrics = iou_metrics * 0.2 + iou_vmetrics * 0.8

            output_valid_message = ""
            output_valid_message += f"\t{'Valid Loss':<20}: {avg_vloss:.6f}\n"
            output_valid_message += f"\t{'Valid IoU':<20}: {iou_vmetrics:.6f}\n"
            output_valid_message += "\n"

            if combined_iou_metrics > iou_record:
                iou_record = combined_iou_metrics
                if not args.no_save:
                    torch.save(model, path_to_save)
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

    path_to_train = os.path.join(os.getcwd(), "data", "train_data")
    path_to_valid = os.path.join(os.getcwd(), "data", "valid_data")

    src_shape = (args.edge_size, args.edge_size)
    # model = build_iterative_model(num_iter=3, num_class=1)
    model = build_model(
        path_to_ckpt=args.checkpoint_dir,
        src_shape=src_shape,
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": model._backbone.parameters(), "lr": args.lr_backbone},
            {"params": model._to_logits.parameters(), "lr": args.lr_backbone},
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if "_backbone" not in name and "_to_logits" not in name
                ]
            },
        ],
        lr=args.learning_rate,
    )

    loss_fn = ModComboLoss(number_of_classes=1, factor=0.3)
    iou_metric = IOUMetric(
        height=src_shape[0], width=src_shape[1], reduction="mean"
    ).to(args.device)

    if args.resume:
        model = torch.load(
            os.path.join(
                args.checkpoint_dir,
                args.resume_from,
            ),
            weights_only=False,
        )

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

    train(
        args,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        data_loader=dataloader,
        epochs=args.epochs,
        valid=True,
        metrics_fn={"iou": iou_metric},
    )
