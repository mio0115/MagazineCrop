from torch.optim.lr_scheduler import ExponentialLR
import torch
import matplotlib.pyplot as plt
from math import log10

from ..utils.misc import move_to_device


def lr_finder(
    model,
    train_loader,
    loss_fn,
    optimizer,
    init_lr=1e-7,
    final_lr=1e-1,
    num_steps=500,
    accumulation_steps: int = 8,
):
    """Implements a simple Learning Rate Finder"""
    model.train()
    lrs, losses = [], []
    gamma = (final_lr / init_lr) ** (1 / num_steps)
    scheduler = ExponentialLR(optimizer, gamma=gamma)

    step_losses = 0
    steps, epoch = 0, 1

    scaler = torch.GradScaler(args.device)
    while True:
        for ind, data in enumerate(train_loader, start=1):
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

            loss = loss / accumulation_steps
            scaler.scale(loss).backward()
            if torch.isnan(loss):
                step_losses += 1.0 / accumulation_steps
            else:
                step_losses += loss.item()

            if ind % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

                for name, param in model.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print(f"Warning: NaN detected in gradients of {name}.")
                        break

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                scheduler.step()

                losses.append(step_losses)
                step_losses = 0

                lrs.append(optimizer.param_groups[0]["lr"])

                steps += 1
                if steps == num_steps:
                    return lrs, losses
        print(f"Epoch {epoch} Step {steps}")
        epoch += 1


if __name__ == "__main__":
    import os

    import torch
    from torch.utils.data import DataLoader

    from .model.model_unet3p import build_model
    from .datasets import MagazineCropDataset, mc_collate_fn
    from .transforms import (
        build_scanned_transform,
    )
    from ..utils.arg_parser import get_parser
    from .loss import ComboLoss

    parser = get_parser("dev")
    args = parser.parse_args()

    src_shape = (args.edge_size, args.edge_size)

    train_dataset = MagazineCropDataset(
        split="train",
        transforms=build_scanned_transform(split="train", reshape_size=args.edge_size),
        augment_factor=args.augment_factor,
    )
    dataloader = {
        "train": DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.dataloader_workers,
            collate_fn=mc_collate_fn,
        ),
    }
    loss_fn = ComboLoss(number_of_classes=1)

    model = build_model(in_channels=3, out_channels=1)
    model = model.to(args.device)

    param_count = sum(p.numel() for p in model.parameters())
    if param_count < 10**6:  # Small model
        init_lr = 1e-6
        final_lr = 1e-2
    elif param_count < 10**8:  # Medium model
        init_lr = 1e-7
        final_lr = 1e-1
    else:  # Large model
        init_lr = 1e-8
        final_lr = 1e-1
    num_steps = int(log10(final_lr / init_lr)) * 50

    # For rest part of model
    for param in model.parameters():
        param.requires_grad = True
    for param in model.encoders.parameters():
        param.requires_grad = False
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if "encoders" not in name
                ]
            },
        ],
        lr=init_lr,
        eps=1e-5,
    )
    # Run LR Finder
    lrs, losses = lr_finder(
        model=model,
        train_loader=dataloader["train"],
        loss_fn=loss_fn,
        init_lr=init_lr,
        final_lr=final_lr,
        num_steps=num_steps,
        optimizer=optimizer,
        accumulation_steps=args.accumulation_steps,
    )
    plt.plot(lrs, losses)
    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("Learning Rate Finder - Head")

    save_path = os.path.join("/", "home", "daniel", "Pictures", "lr_finder_head.pdf")
    plt.savefig(save_path, dpi=600, format="pdf")
    plt.clf()

    # For backbone
    model = build_model(in_channels=3, out_channels=1)
    model = model.to(args.device)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.encoders.parameters():
        param.requires_grad = True
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoders.parameters()},
        ],
        lr=init_lr,
        eps=1e-5,
    )
    lrs, losses = lr_finder(
        model=model,
        train_loader=dataloader["train"],
        loss_fn=loss_fn,
        init_lr=init_lr,
        final_lr=final_lr,
        num_steps=num_steps,
        optimizer=optimizer,
        accumulation_steps=args.accumulation_steps,
    )
    plt.plot(lrs, losses)
    plt.xscale("log")
    plt.xlabel("Backbone Learning Rate")
    plt.ylabel("Loss")
    plt.title("Learning Rate Finder - Backbone")
    save_path = os.path.join(
        "/", "home", "daniel", "Pictures", "lr_finder_backbone.pdf"
    )
    plt.savefig(save_path, dpi=600, format="pdf")
