import boto3
import json
import boto3
import json
import os
import logging

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

ENDPOINT_NAME = os.getenv('ENDPOINT_NAME')
runtime = boto3.client('runtime.sagemaker')

def lambda_handler(event, context):
    logger.info(f"Event received: {event}")

    payload = json.dumps(event)

    try:
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='application/json',
            Body=payload
        )

        result = json.loads(response['Body'].read().decode())
        logger.info(f"Inference result: {result}")
        return result

    except Exception as e:
        logger.error(f"Error invoking endpoint: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"Invocation failed: {str(e)}")
        }
