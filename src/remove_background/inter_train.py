import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from .model.mod_unet_pp import InterModel
from .datasets import InterMagazineCropDataset
from .mod_transforms import build_inter_transform
from ..utils.arg_parser import get_parser
from .loss import ComboLoss

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
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15])
    model = model.to(args.device)
    path_to_save = os.path.join(args.checkpoint_dir, args.save_as)

    best_loss = float("inf")
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        for ind, data in enumerate(data_loader["train"]):
            src, tgt, weights, edge_len, edge_theta = data
            src = src.to(args.device)
            tgt = tgt.to(args.device)
            weights = weights.to(args.device)
            edge_len = edge_len.to(args.device)
            edge_theta = edge_theta.to(args.device)

            optimizer.zero_grad()
            logits = model(src, edge_len, edge_theta)

            loss = loss_fn(logits, tgt, weights)
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
                    src, tgt, weights, edge_len, edge_theta = data
                    src = src.to(args.device)
                    tgt = tgt.to(args.device)
                    weights = weights.to(args.device)
                    edge_len = edge_len.to(args.device)
                    edge_theta = edge_theta.to(args.device)

                    logits = model(src, edge_len, edge_theta)
                    loss = loss_fn(logits, tgt, weights)
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

    model = InterModel()
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters()},
        ],
        lr=args.learning_rate,
    )

    loss_fn = ComboLoss(number_of_classes=1)

    train_dataset = InterMagazineCropDataset(
        split="train",
        transforms=build_inter_transform("train"),
        augment_factor=args.augment_factor,
    )
    valid_dataset = InterMagazineCropDataset(
        split="valid", transforms=build_inter_transform("valid"), augment_factor=2
    )
    dataloader = {
        "train": DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.dataloader_workers,
            pin_memory=True,
        ),
        "valid": DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.dataloader_workers,
            pin_memory=True,
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
