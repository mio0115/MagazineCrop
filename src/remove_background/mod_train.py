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
):
    print("Training model...")
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15])
    model = model.to(args.device)
    path_to_save = os.path.join(args.checkpoint_dir, args.save_as)

    best_loss = float("inf")
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
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

            loss = loss_fn(outputs=outputs, targets=targets, weights=weights)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

        scheduler.step()
        avg_loss = running_loss / (ind + 1)
        print(f"Epoch {epoch+1:>2}:\n\t{'Train Loss':<11}: {avg_loss:.6f}")

        if valid:
            model.eval()
            with torch.no_grad():
                running_vloss = 0.0
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

                    loss = loss_fn(outputs=outputs, targets=targets, weights=weights)

                    running_vloss += loss.item()

                avg_vloss = running_vloss / (ind + 1)

                output_avg_vloss = f"\t{'Valid Loss':<11}: {avg_vloss:.6f}\n"
                output_avg_vloss += "\n"

                if avg_vloss < best_loss:
                    best_loss = avg_vloss
                    torch.save(model, path_to_save)
                    output_avg_vloss += "\tNew Record, Saved!"
                print(output_avg_vloss)
                print(f"\t{'Best Loss':<11}: {best_loss:.6f}")

        epoch_end = time.time()
        min_t = (epoch_end - epoch_start) // 60
        sec_t = int((epoch_end - epoch_start) % 60)
        print(f"\t{'Epoch Time':<11}: {min_t} min(s) {sec_t} sec(s)\n")


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
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if "_backbone" not in name
                ]
            },
        ],
        lr=args.learning_rate,
    )

    loss_fn = ModComboLoss(number_of_classes=1)

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
    )
