import time

from .pipeline.pipeline import MagazineCropPipeline


def main():
    start_time = time.time()

    pipeline = MagazineCropPipeline(parser_mode="user")
    pipeline()

    end_time = time.time()
    print(f"Completed in {end_time-start_time:.2f} seconds")


if __name__ == "__main__":
    main()
