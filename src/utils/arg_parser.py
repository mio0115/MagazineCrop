import argparse


def get_parser(mode: str = "user"):
    if mode == "user":
        return user_parser()
    elif mode == "dev":
        return dev_parser()
    elif mode == "output":
        return output_parser()
    else:
        raise ValueError(
            f"Invalid mode; mode should be either 'user' or 'dev' but got {mode}"
        )


def user_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input-image",
        type=str,
        required=True,
        dest="input",
        help="Path to the input image",
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        type=str,
        required=True,
        dest="output_dir",
        help="Path to the directory where output files will be saved",
    )
    parser.add_argument(
        "--scale-factor",
        "--output-scale",
        type=float,
        default=1.0,
        dest="output_scale",
        help="Scale area for the output images; default is 1.0",
    )
    parser.add_argument(
        "-q",
        "--compression-quality",
        type=int,
        default=95,
        dest="quality",
        help="Compression quality for the output images",
    )
    parser.add_argument(
        "--use-gpu", action="store_true", dest="use_gpu", help="Enable GPU usage"
    )
    parser.add_argument(
        "--single-page",
        action="store_true",
        dest="single_page",
        help="Input image has only one page",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        dest="verbose",
        help="Set verbosity level",
    )

    return parser


def dev_parser():
    parser = argparse.ArgumentParser()

    # arguments for training model
    training_group = parser.add_argument_group("Training arguments")
    training_group.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=10,
        dest="epochs",
        help="number of epochs to train the model",
    )
    training_group.add_argument(
        "-bs",
        "--batch-size",
        type=int,
        default=12,
        dest="batch_size",
        help="batch size for training",
    )
    training_group.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=1e-3,
        dest="learning_rate",
        help="learning rate for training",
    )
    training_group.add_argument(
        "--lr-backbone",
        type=float,
        default=1e-4,
        dest="lr_backbone",
        help="learning rate for backbone",
    )
    training_group.add_argument(
        "--device",
        type=str,
        default="cuda",
        dest="device",
        help="device to train the model on",
    )
    training_group.add_argument(
        "-ckpt-dir",
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        dest="checkpoint_dir",
        help="path to save the model",
    )
    training_group.add_argument(
        "--save-as",
        type=str,
        default="rm_bg_unetpp",
        dest="save_as",
        help="store the model in path/to/checkpoints/save_as/save_as.pth",
    )
    training_group.add_argument(
        "--augment-factor", type=int, default=5, dest="augment_factor"
    )
    training_group.add_argument(
        "--train", action="store_true", dest="train", help="train the model"
    )
    training_group.add_argument(
        "--dataloader-workers",
        type=int,
        default=3,
        dest="dataloader_workers",
        help="number of workers for dataloader",
    )
    training_group.add_argument("--resume", action="store_true", dest="resume")
    training_group.add_argument(
        "--resume-from",
        type=str,
        default="rm_bg_unetpp_pretrained",
        dest="resume_from",
        help="resume training from path/to/checkpoint/resume_from/resume_from.pth",
    )
    training_group.add_argument(
        "--edge-size",
        type=int,
        default=640,
        choices=[480, 640, 1024],
        dest="edge_size",
        help="height and width for src",
    )
    training_group.add_argument(
        "--no-save",
        action="store_true",
        dest="no_save",
        help="do not save model after training",
    )
    training_group.add_argument(
        "-accum-steps",
        "--accumulation-steps",
        type=int,
        default=1,
        dest="accumulation_steps",
        help="number of steps of gradient accumulation",
    )
    training_group.add_argument(
        "-mp",
        "--mixed-precision",
        action="store_true",
        dest="mixed_precision",
        help="train with mixed precision",
    )

    # arguments for inference
    inference_group = parser.add_argument_group("Inference arguments")
    inference_group.add_argument(
        "--image-dir",
        type=str,
        default="C2980",
        dest="image_dir",
        help="name of the image directory to inference",
    )
    inference_group.add_argument(
        "--image-name", type=str, default="no1-0818_132552.tiff", dest="image_name"
    )
    inference_group.add_argument(
        "--output-dir",
        type=str,
        default="./data/output",
        dest="output_dir",
        help="path to the output directory",
    )
    inference_group.add_argument(
        "--output-name", type=str, default="image.jpg", dest="output_as"
    )
    inference_group.add_argument(
        "--rm-bg-model",
        type=str,
        dest="rm_bg_model",
        help="path to remove-background model. \
            For example, ./src/remove_background/checkpoints/rm_bg_model.pth",
    )
    inference_group.add_argument(
        "--sp-pg-model",
        type=str,
        dest="sp_pg_model",
        help="path to split-page model. \
            For example, ./src/split_page/checkpoints/sp_pg_model.pth",
    )

    parser.add_argument(
        "-v", "--verbose", action="count", default=0, dest="Set verbosity level"
    )
    parser.add_argument(
        "--num-pages", type=int, default=2, choices=[1, 2], dest="num_pages"
    )

    return parser


def output_parser():
    parser = argparse.ArgumentParser(
        description="Combine the original and processed images into a single output."
    )
    parser.add_argument(
        "--original-image",
        type=str,
        required=True,
        dest="original",
        help="Path to the original image",
    )
    parser.add_argument(
        "--processed-images",
        type=str,
        nargs="+",
        dest="processed",
        help="Path to the processed images. The order should be left, right.",
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        type=str,
        required=True,
        dest="output_dir",
        help="Path to the output directory",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        dest="save_as",
        help="Save the output as a specific name. Default name is the same as original image.",
    )
    parser.add_argument(
        "--scale-factor",
        "--output-scale",
        type=float,
        default=1.0,
        dest="scale",
        help="Scale of the output image. Default is 1.0.",
    )
    parser.add_argument(
        "--original-scale-factor", type=float, default=1.0, dest="original_scale_factor"
    )

    return parser
