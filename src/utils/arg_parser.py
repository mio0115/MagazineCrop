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
        "--backbone_lr",
        type=float,
        default=1e-4,
        dest="backbone_lr",
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
        "--path_to_train_data",
        type=str,
        default="./data/train_data",
        dest="path_to_train_data",
        help="path to the training dataset",
    )
    parser.add_argument(
        "--path_to_valid_data",
        type=str,
        default="./data/valid_data",
        dest="path_to_valid_data",
        help="path to the validation dataset",
    )
    parser.add_argument("--augment_factor", type=int, default=5, dest="augment_factor")
    parser.add_argument(
        "--train", type=bool, action="store_true", dest="train", help="train the model"
    )

    # arguments for inference
    parser.add_argument(
        "--path_to_image_dir",
        type=str,
        default="./data/example/no1-0818_132552.tiff",
        dest="path_to_image_dir",
        help="path to the input image directory",
    )
    parser.add_argument(
        "--image_name", type=str, default="image.tiff", dest="image_name"
    )
    parser.add_argument(
        "--path_to_model_dir",
        type=str,
        default="./remove_background/checkpoints",
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

    return parser
