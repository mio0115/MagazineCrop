from PIL import Image
import pyheif


def convert_heic_to(image_path, output_path, output_format="JPEG"):
    # READ HEIC IMAGE
    heif_file = pyheif.read(image_path)

    # Convert to PIL Image
    image = Image.frombytes(
        heif_file.mode,
        heif_file.size,
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride,
    )

    image.save(output_path, format=output_format)


if __name__ == "__main__":
    import os

    convert_heic_to(
        image_path=os.path.join("/", "home", "daniel", "Downloads", "IMG_0020.HEIC"),
        output_path=os.path.join("/", "home", "daniel", "Downloads", "IMG_0020.jpg"),
        output_format="JPEG",
    )
