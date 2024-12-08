import argparse


def get_parser():
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
        "--batch_size",
        type=int,
        default=12,
        dest="batch_size",
        help="batch size for training",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=1e-3,
        dest="learning_rate",
        help="learning rate for training",
    )
    parser.add_argument(
        "--lr_backbone",
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
        "--save_as", type=str, default="rm_bg_unetpp.pth", dest="save_as"
    )
    parser.add_argument("--augment_factor", type=int, default=5, dest="augment_factor")
    parser.add_argument(
        "--train", action="store_true", dest="train", help="train the model"
    )
    parser.add_argument("--resume", action="store_true", dest="resume")
    parser.add_argument(
        "--resume_from",
        type=str,
        default="rm_bg_unetpp_pretrained.pth",
        dest="resume_from",
    )

    # arguments for inference
    parser.add_argument(
        "--path_to_image_dir",
        type=str,
        default="./data/example",
        dest="path_to_image_dir",
        help="path to the input image directory",
    )
    parser.add_argument(
        "--image_name", type=str, default="no1-0818_132552.tiff", dest="image_name"
    )
    parser.add_argument(
        "--path_to_model_dir",
        type=str,
        default="./src/remove_background/checkpoints",
        dest="path_to_model_dir",
        help="path to the directory placed trained model",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="model.pth",
        dest="model_name",
        help="name of the model",
    )
    parser.add_argument(
        "--path_to_output_dir",
        type=str,
        default="./data/output",
        dest="path_to_output_dir",
        help="path to the output directory",
    )
    parser.add_argument("--output_as", type=str, default="image.jpg", dest="output_as")
    parser.add_argument("--rm-bg-model-name", type=str, default="rm_bg_entire_iter.pth")
    parser.add_argument("--sp-pg-model-name", type=str, default="sp_pg_mod.pth")

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
    parser.add_argument("--output-scale", type=float, default=1.0, dest="output_scale")
    parser.add_argument("--compression-quality", type=int, default=95, dest="quality")
    parser.add_argument(
        "--save-steps-output", action="store_true", dest="save_steps_output"
    )
    parser.add_argument(
        "--num-pages", type=int, default=2, choices=[1, 2], dest="num_pages"
    )

    return parser
