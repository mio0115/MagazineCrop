from torch.optim.lr_scheduler import ExponentialLR
import torch
import matplotlib.pyplot as plt
from itertools import cycle


def move_to_device(
    data: dict[str, torch.Tensor] | torch.Tensor, device: str | torch.device
) -> dict[str, torch.Tensor]:
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: v.to(device) for k, v in data.items()}
    else:
        raise ValueError("Data should be either a dict or a tensor")


def lr_finder(
    model,
    train_loader,
    loss_fn,
    optimizer,
    init_lr=1e-7,
    final_lr=1e-2,
    num_steps=800,
    accumulation_steps: int = 8,
    batch_size: int = 2,
):
    """Implements a simple Learning Rate Finder"""
    model.train()
    lrs, losses, backbone_lrs = [], [], []
    gamma = (final_lr / init_lr) ** (1 / num_steps)
    scheduler = ExponentialLR(optimizer, gamma=gamma)

    step_losses = 0
    for i, data in enumerate(cycle(train_loader)):
        if i >= num_steps:
            break

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
        loss = loss / accumulation_steps
        loss.backward()

        step_losses += loss.item()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            losses.append(step_losses)
            step_losses = 0

            backbone_lrs.append(optimizer.param_groups[0]["lr"])
            lrs.append(optimizer.param_groups[-1]["lr"])

    return lrs, backbone_lrs, losses


if __name__ == "__main__":
    import os

    import torch
    from torch.utils.data import DataLoader

    from .model.mod_unet_pp import build_model
    from .datasets import ModMagazineCropDataset, mod_mc_collate_fn
    from .mod_transforms import (
        build_scanned_transform,
    )
    from ..utils.arg_parser import get_parser
    from .loss import ModComboLoss
    from .metrics import IOUMetric

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
    model = model.to(args.device)

    loss_fn = ModComboLoss(number_of_classes=1, factor=0.3)
    iou_metric = IOUMetric(
        height=src_shape[0], width=src_shape[1], reduction="mean"
    ).to(args.device)

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

    # Run LR Finder
    lrs, backbone_lrs, losses = lr_finder(
        model,
        dataloader["train"],
        loss_fn,
        optimizer,
        accumulation_steps=args.accumulation_steps,
        batch_size=args.batch_size,
    )
    print("LEARNING RATE")
    for lr, loss in zip(lrs, losses):
        print(f"learning rate: {lr:.6f}, loss: {loss:.6f}")
    print("BACKBONE LEARNING RATE")
    for lr, loss in zip(backbone_lrs, losses):
        print(f"learning rate: {lr:.6f}, loss: {loss:.6f}")
    plt.plot(lrs, losses)
    # plt.plot(backbone_lrs, losses)
    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("Learning Rate Finder")
    plt.show()
