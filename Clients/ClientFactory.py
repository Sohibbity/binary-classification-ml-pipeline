from functools import cached_property

import boto3
from botocore.client import BaseClient


class ClientFactory:
    """
    Manual Dependency Injection
    Can move to actual DI framework as project expands
    For now this supports lazy init of dependenices and is more than sufficient
    """
    def __init__(self):
        pass

    @cached_property
    def s3_client(self):
        return boto3.client('s3')

    @cached_property
    def sagemaker_client(self):
        return boto3.client('sagemaker', region_name = 'us-east-2')
