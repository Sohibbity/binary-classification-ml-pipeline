import boto3
import json

# Configuration
ENDPOINT_NAME = 'binary-classifier-endpoint'
REGION = 'us-east-2'


def test_endpoint():
    """
    Temp testing script to debug sagemaker inference endpoint
    """

    # Create runtime client
    runtime_client = boto3.client('sagemaker-runtime', region_name=REGION)

    # Sample input: 3 rows of 8 features each
    sample_data = [
        [30, 1787, 1, -1, 0, 0, 0, 0],
        [33, 4789, 1, 339, 4, 0, 1, 1],
        [35, 1350, 1, 330, 1, 0, 1, 0]
    ]

    # Serialize to JSON
    payload = json.dumps(sample_data)

    print(f"Calling endpoint: {ENDPOINT_NAME}")
    print(f"Payload: {payload}")

    # Invoke endpoint
    response = runtime_client.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType='application/json',
        Accept='application/json',
        Body=payload
    )

    # Parse response
    result = json.loads(response['Body'].read().decode())
    print(f"\nResponse:")
    print(json.dumps(result, indent=2))

    return result


if __name__ == "__main__":
    test_endpoint()