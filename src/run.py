import time

from .pipeline.pipeline import MagazineCropPipeline, load_image


def main():
    start_time = time.time()

    pipeline = MagazineCropPipeline(parser_mode="user")

    pipeline()

    # args = pipeline.get_args()
    # pipeline.valid_and_set(args=args)
    # pipeline.init_pipeline()

    # image = load_image(path=pipeline._input_path)
    # fixed_pages = pipeline.process(image=image)
    # pipeline.save_output(fixed_pages=fixed_pages)

    end_time = time.time()
    print(f"Completed in {end_time-start_time:.2f} seconds")


if __name__ == "__main__":
    main()
