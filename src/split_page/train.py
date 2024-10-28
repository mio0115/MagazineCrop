import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from .model.model import build_model
from .datasets import MagazineCropDataset
from .transforms import build_scanned_transforms

from ..utils.arg_parser import get_parser
from .loss import MeanSquaredLoss

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
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25, 30])
    model = model.to(args.device)
    path_to_save = os.path.join(os.getcwd(), "checkpoints", args.save_as)

    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for ind, data in enumerate(data_loader["train"]):
            src, tgt = data
            src = src.to(args.device)
            tgt = tgt.to(args.device)

            optimizer.zero_grad()
            logits = model(src)

            loss = loss_fn(logits, tgt)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

        scheduler.step()
        avg_loss = running_loss / (ind + 1)
        print(f"Epoch {epoch+1:>2}:\n\t{'Train Loss':<11}: {avg_loss:.4f}")

        if valid:
            model.eval()
            with torch.no_grad():
                running_vloss = 0.0
                for ind, data in enumerate(data_loader["valid"]):
                    src, tgt = data
                    src = src.to(args.device)
                    tgt = tgt.to(args.device)

                    logits = model(src)
                    loss = loss_fn(logits, tgt)

                    running_vloss += loss.item()

                avg_vloss = running_vloss / (ind + 1)

                output_avg_vloss = f"\t{'Valid Loss':<11}: {avg_vloss:.4f}\n"

                if avg_vloss < best_loss:
                    best_loss = avg_vloss
                    torch.save(model.state_dict(), path_to_save)
                    output_avg_vloss += "\tNew best loss, Saved!"
                print(output_avg_vloss)
                print(f"\t{'Best Loss':<11}: {best_loss:.4f}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    path_to_train = os.path.join(os.getcwd(), "data", "train_data")
    path_to_valid = os.path.join(os.getcwd(), "data", "valid_data")

    model = build_model()
    optimizer = torch.optim.AdamW(
        [
            {"params": model._backbone.parameters(), "lr": args.lr_backbone},
            {"params": model._x_reg_head.parameters()},
            {"params": model._theta_reg_head.parameters()},
        ],
        lr=args.learning_rate,
    )
    loss_fn = MeanSquaredLoss()

    if args.resume:
        model.load_state_dict(
            torch.load(
                os.path.join(
                    os.getcwd(),
                    "checkpoints",
                    args.resume_from,
                ),
                weights_only=True,
            )
        )

    train_dataset = MagazineCropDataset(
        split="train", transforms=build_scanned_transforms()
    )
    valid_dataset = MagazineCropDataset(
        split="valid", transforms=build_scanned_transforms()
    )
    dataloader = {
        "train": DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(os.cpu_count() - 1, 0),
        ),
        "valid": DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=max(os.cpu_count() - 1, 0),
        ),
    }

    train(
        args,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        data_loader=dataloader,
        epochs=args.epochs,
    )
