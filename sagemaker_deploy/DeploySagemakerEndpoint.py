from Clients.ClientFactory import ClientFactory

# Configuration
# TODO - extract to unified constants file, being scrappy in the time being.

MODEL_DATA = 's3://ml-inference-data-soheeb/models/model.tar.gz'
REGION = 'us-east-1'
INSTANCE_TYPE = 'ml.t2.medium'
CONTAINER_IMAGE = '763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-inference:2.0.0-cpu-py310'

# Unique names (add timestamp to avoid conflicts)
MODEL_NAME = f'binary-classifier'
ENDPOINT_CONFIG_NAME = f'binary-classifier-config'
ENDPOINT_NAME = f'binary-classifier-endpoint'

# Initialize client
client_factory = ClientFactory()
sagemaker_client = client_factory.sagemaker_client


def create_model():
    """
    Creates a SageMaker Model that points to your model artifacts and container image.
    """
    print(f"Creating SageMaker Model: {MODEL_NAME}")

    response = sagemaker_client.create_model(
        ModelName=MODEL_NAME,
        PrimaryContainer={
            'Image': CONTAINER_IMAGE,
            'ModelDataUrl': MODEL_DATA,
            'Environment': {
                'SAGEMAKER_PROGRAM': 'inference.py',
                'SAGEMAKER_SUBMIT_DIRECTORY': MODEL_DATA,
                'SAGEMAKER_REGION': REGION
            }
        },
        ExecutionRoleArn=ROLE_ARN
    )

    print(f"Model ARN: {response['ModelArn']}")
    return response


def create_endpoint_config():
    """
    Creates an Endpoint Configuration specifying instance type and count.
    """
    print(f"Creating Endpoint Configuration: {ENDPOINT_CONFIG_NAME}")

    response = sagemaker_client.create_endpoint_config(
        EndpointConfigName=ENDPOINT_CONFIG_NAME,
        ProductionVariants=[
            {
                'VariantName': 'AllTraffic',
                'ModelName': MODEL_NAME,
                'InstanceType': INSTANCE_TYPE,
                'InitialInstanceCount': 1,
            }
        ]
    )

    print(f"Endpoint Config ARN: {response['EndpointConfigArn']}")
    return response


def create_endpoint():
    """
    Creates and deploys the endpoint. This provisions infrastructure and starts serving.
    """
    print(f"Creating Endpoint: {ENDPOINT_NAME}")
    print("This will take 5-10 minutes...")

    response = sagemaker_client.create_endpoint(
        EndpointName=ENDPOINT_NAME,
        EndpointConfigName=ENDPOINT_CONFIG_NAME
    )

    print(f"Endpoint ARN: {response['EndpointArn']}")

    # Wait for endpoint to be in service
    print("Waiting for endpoint to be InService...")
    waiter = sagemaker_client.get_waiter('endpoint_in_service')
    waiter.wait(EndpointName=ENDPOINT_NAME)

    print(f"✅ Endpoint {ENDPOINT_NAME} is now InService!")
    return response


def main():
    # Model only needs to be run once per unique model to store metadata in Sagemaker.
    # create_model()
    create_endpoint_config()
    create_endpoint()

if __name__ == '__main__':
    main()