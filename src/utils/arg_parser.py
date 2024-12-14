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
        "--input",
        type=str,
        required=True,
        dest="input",
        help="Path to the input image",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        dest="output_dir",
        help="Path to the directory where output files will be saved",
    )
    parser.add_argument(
        "--output-scale",
        type=float,
        default=1.0,
        dest="output_scale",
        help="Scale area for the output images; default is 1.0",
    )
    parser.add_argument(
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
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=10,
        dest="epochs",
        help="number of epochs to train the model",
    )
    parser.add_argument(
        "-bs",
        "--batch-size",
        type=int,
        default=12,
        dest="batch_size",
        help="batch size for training",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=1e-3,
        dest="learning_rate",
        help="learning rate for training",
    )
    parser.add_argument(
        "--lr-backbone",
        type=float,
        default=1e-4,
        dest="lr_backbone",
        help="learning rate for backbone",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        dest="device",
        help="device to train the model on",
    )
    parser.add_argument(
        "--save-as", type=str, default="rm_bg_unetpp.pth", dest="save_as"
    )
    parser.add_argument("--augment_factor", type=int, default=5, dest="augment_factor")
    parser.add_argument(
        "--train", action="store_true", dest="train", help="train the model"
    )
    parser.add_argument("--resume", action="store_true", dest="resume")
    parser.add_argument(
        "--resume-from",
        type=str,
        default="rm_bg_unetpp_pretrained.pth",
        dest="resume_from",
    )

    # arguments for inference
    parser.add_argument(
        "--path-to-image_dir",
        type=str,
        default="./data/example",
        dest="path_to_image_dir",
        help="path to the input image directory",
    )
    parser.add_argument(
        "--image-name", type=str, default="no1-0818_132552.tiff", dest="image_name"
    )
    parser.add_argument(
        "--path-to-model-dir",
        type=str,
        default="./src/remove_background/checkpoints",
        dest="path_to_model_dir",
        help="path to the directory placed trained model",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="model.pth",
        dest="model_name",
        help="name of the model",
    )
    parser.add_argument(
        "--path-to-output-dir",
        type=str,
        default="./data/output",
        dest="path_to_output_dir",
        help="path to the output directory",
    )
    parser.add_argument("--output-as", type=str, default="image.jpg", dest="output_as")
    parser.add_argument("--rm-bg-model-name", type=str, default="rm_bg_entire_iter.pth")
    parser.add_argument("--sp-pg-model-name", type=str, default="sp_pg_mod.pth")

    parser.add_argument(
        "-v", "--verbose", action="count", default=0, dest="Set verbosity level"
    )
    parser.add_argument(
        "--num-pages", type=int, default=2, choices=[1, 2], dest="num_pages"
    )

    return parser


def output_parser():
    parser = argparse.ArgumentParser(
        description="Combine the original and processed images."
    )
    parser.add_argument(
        "--original", type=str, required=True, help="Path to the original image"
    )
    parser.add_argument(
        "--processed",
        type=str,
        nargs="+",
        help="Path to the processed images. The order should be left, right.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        dest="output_dir",
        help="Path to the output directory",
    )
    parser.add_argument(
        "--save-as",
        type=str,
        dest="save_as",
        help="Save the output as a specific name. Default name is the same as original image.",
    )
    parser.add_argument(
        "--scale",
        "--output-scale",
        type=float,
        default=1.0,
        dest="scale",
        help="Scale of the output image. Default is 1.0.",
    )

    return parser
