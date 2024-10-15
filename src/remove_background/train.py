import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from .model.model_unet_pp import build_unetplusplus
from .datasets import MyVOCSegmentation
from .transforms import build_transforms, build_valid_transform
from ..utils.arg_parser import get_parser
from .loss import ComboLoss


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
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15, 20])
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
        print(f"Epoch: {epoch}:\n\tTrain Loss: {avg_loss:.4f}")

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
                output_avg_vloss = f"\n\tValid Loss: {avg_vloss:.4f}"
                if avg_vloss < best_loss:
                    best_loss = avg_vloss
                    torch.save(model.state_dict(), path_to_save)
                    output_avg_vloss += ", Saved!"
                print(output_avg_vloss)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    path_to_train = os.path.join(os.getcwd(), "data", "train_data")
    path_to_valid = os.path.join(os.getcwd(), "data", "valid_data")

    model = build_unetplusplus(number_of_classes=1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )
    loss_fn = ComboLoss(number_of_classes=1)
    train_dataset = MyVOCSegmentation(
        root=path_to_train,
        image_set="train",
        transforms=build_transforms(args),
        augment_factor=args.augment_factor,
    )
    valid_dataset = MyVOCSegmentation(
        root=path_to_valid,
        image_set="val",
        transforms=build_valid_transform(args),
        augment_factor=1,
    )
    dataloader = {
        "train": DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True),
        "valid": DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False),
    }

    train(
        args,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        data_loader=dataloader,
        epochs=args.epochs,
    )
