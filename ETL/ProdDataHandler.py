import io
from pathlib import Path

from botocore.response import StreamingBody
from pandas import DataFrame

from Clients.ClientFactory import ClientFactory


class ProdDataHandler:
    def __init__(self, client_factory: ClientFactory):
        self.s3_client = client_factory.s3_client

    def stream_inference_input_file(self, bucket: str, key: str) -> StreamingBody:
        """
        Creates an S3 object stream
        Avoids loading entire file into disk
        Use for production workloads/datasets
        """
        obj = self.s3_client.get_object(Bucket = bucket, Key = key)

        return obj['Body']

    def load_inference_input_file(self, bucket: str, key: str, local_path: str):
        """
        Loads entire file onto disk
        Use for Local debugging/smaller datasets
        """
        self.s3_client.download_file(bucket, key , local_path)

    # Defensability Analysis:
    # why would write to s3 fail? - these are 99.9999% time due to network issues
    # i.e con dropped/timeout.
    # simply retry on netowrk issues
    def stream_write_file(self, bucket:str, key: str, df: DataFrame, chunk_id : int):
        """
        Writes a single chunk to S3.
        key_prefix: 's3://bucket/predictions/2024-11-17/run-12345/'
        chunk_id: 0, 1, 2, ...
        """

        # Convert DF to writable CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=True)
        df_as_csv = csv_buffer.getvalue()

        # Create unique key for this chunk
        # key = f"{key}/chunk_{chunk_id}.csv"
        key = f"{key}/chunk_{chunk_id:06d}.csv"  # No extra directory!

        # Write File
        # Note s3 files are immutable - upload each chunk as a separate file

        self.s3_client.put_object(Bucket = bucket, Key = key, Body = df_as_csv )