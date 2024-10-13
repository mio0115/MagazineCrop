import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from .model.model_unet_pp import build_unetplusplus
from .datasets import MyVOCSegmentation
from .transforms import build_transforms, build_valid_transform
from ..utils.arg_parser import get_parser
from .loss import MultiClassDiceLoss


def train(
    args,
    model: nn.Module,
    optimizer,
    loss_fn: nn.Module,
    data_loader: dict[str, DataLoader],
    epochs: int,
    valid: bool = True,
):
    model = model.to(args.device)

    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        for data in data_loader["train"]:
            src, tgt = data
            src = src.to(args.device)
            tgt = tgt.to(args.device)

            optimizer.zero_grad()
            logits = model(src)

            loss = loss_fn(logits, tgt)
            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch}, Loss: {loss.item()}")
        if valid:
            model.eval()
            with torch.no_grad():
                vloss = 0.0
                for ind, data in enumerate(data_loader["valid"]):
                    src, tgt = data
                    src = src.to(args.device)
                    tgt = tgt.to(args.device)

                    logits = model(src)
                    loss = loss_fn(logits, tgt)
                    vloss += loss.item()
                    break

                vloss /= ind + 1
                if vloss < best_loss:
                    best_loss = vloss
                    torch.save(
                        model.state_dict(),
                        os.path.join(
                            os.getcwd(),
                            "src",
                            "remove_background",
                            "checkpoints",
                            args.save_as,
                        ),
                    )
                    print(
                        f"Epoch: {epoch}, Loss: {loss.item()}, Best Loss: {best_loss}"
                    )
                else:
                    print(f"Epoch: {epoch}, Loss: {loss.item()}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    path_to_train = os.path.join(os.getcwd(), "data", "train_data")
    path_to_valid = os.path.join(os.getcwd(), "data", "valid_data")

    model = build_unetplusplus()
    optimizer = torch.optim.AdamW(
        [
            {"params": model._contract_blk.parameters(), "lr": args.backbone_lr},
            {"params": model._expand_blks.parameters(), "lr": args.backbone_lr},
        ],
        lr=args.learning_rate,
    )
    loss_fn = MultiClassDiceLoss(ignore_index=20)
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
        "valid": DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True),
    }

    train(
        args,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        data_loader=dataloader,
        epochs=args.epochs,
    )
