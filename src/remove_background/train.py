import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

# from .model.model_unet import build_iterative_model
from .model.model_unet_pp import build_iterative_model
from .datasets import MyVOCSegmentation, MagazineCropDataset
from .transforms import (
    build_transform,
    build_valid_transform,
    build_scanned_transform,
)
from ..utils.arg_parser import get_parser
from .loss import ComboLoss
from .metrics import BinaryMetrics

# to download model's weights, execute the following command:
# scp <username>@<ip>:/home/ubuntu/projects/MagazineCrop/src/remove_background/checkpoints/<model_name> ./src/remove_background/checkpoints/


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
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 25, 30])
    model = model.to(args.device)
    path_to_save = os.path.join(args.path_to_save, args.save_as)

    best_loss = float("inf")
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        for ind, data in enumerate(data_loader["train"]):
            src, tgt, weights = data
            src = src.to(args.device)
            tgt = tgt.to(args.device)
            weights = weights.to(args.device)

            optimizer.zero_grad()
            logits = model(src)

            loss = loss_fn(logits, tgt, weights)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

        scheduler.step()
        avg_loss = running_loss / (ind + 1)
        print(f"Epoch {epoch+1:>2}:\n\t{'Train Loss':<11}: {avg_loss:.6f}")

        metrics = {
            "pixel_acc": 0.0,
            "dice": 0.0,
            "precision": 0.0,
            "specificity": 0.0,
            "recall": 0.0,
        }

        if valid:
            model.eval()
            with torch.no_grad():
                running_vloss = 0.0
                for ind, data in enumerate(data_loader["valid"]):
                    src, tgt, weights = data
                    src = src.to(args.device)
                    tgt = tgt.to(args.device)
                    weights = weights.to(args.device)

                    logits = model(src)
                    loss = loss_fn(logits, tgt, weights)
                    # new_metrics: dict[str, torch.Tensor] = BinaryMetrics()(
                    #     y_pred=logits, y_true=tgt
                    # )

                    # for key in new_metrics.keys():
                    #     metrics[key] += new_metrics[key].item()

                    running_vloss += loss.item()

                avg_vloss = running_vloss / (ind + 1)
                # for key in metrics.keys():
                #     metrics[key] /= ind + 1

                output_avg_vloss = f"\t{'Valid Loss':<11}: {avg_vloss:.6f}\n"
                # for key in metrics.keys():
                #     output_avg_vloss += f"\t{key:<11}: {metrics[key]:.6f}\n"
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

    # model = build_iterative_model(num_iter=3, num_class=1)
    model = build_iterative_model()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )
    loss_fn = ComboLoss(number_of_classes=1)

    if args.resume:
        model = torch.load(
            os.path.join(
                os.getcwd(),
                "checkpoints",
                args.resume_from,
            ),
            weights_only=False,
        )

    # train_dataset = MyVOCSegmentation(
    #     root=path_to_train,
    #     image_set="train",
    #     transforms=build_transforms(args),
    #     augment_factor=args.augment_factor,
    # )
    # valid_dataset = MyVOCSegmentation(
    #     root=path_to_valid,
    #     image_set="val",
    #     transforms=build_valid_transform(args),
    #     augment_factor=1,
    # )
    train_dataset = MagazineCropDataset(
        split="train",
        transforms=build_scanned_transform(),
        augment_factor=args.augment_factor,
    )
    valid_dataset = MagazineCropDataset(
        split="valid", transforms=build_valid_transform(), augment_factor=2
    )
    dataloader = {
        "train": DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.dataloader_workers,
        ),
        "valid": DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.dataloader_workers,
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
