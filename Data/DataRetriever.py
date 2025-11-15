
from shutil import copyfile

class DataRetriever:
    """
    Currently loads from a locally stored csv file.
    Production ready would load from a blob storage source (i.e AWS S3)
    """

    @staticmethod
    def load_data(source_file_path: str, output_file_path: str):
        copyfile(source_file_path, output_file_path)
