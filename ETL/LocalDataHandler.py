from pathlib import Path
from shutil import copyfile

from Config.Constants import INPUT_DATA_DIR


class LocalDataHandler:
    """
    Currently loads from a locally stored csv file.
    Production ready would load from a blob storage source (i.e AWS S3)
    """

    @staticmethod
    def load_data(source_file_path: Path, output_file_path: Path):
        destination = INPUT_DATA_DIR / output_file_path
        copyfile(source_file_path, destination)
