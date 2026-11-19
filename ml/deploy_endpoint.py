"""Create (or replace) the SageMaker Serverless Inference endpoint for the forecast model.

Uses the built-in XGBoost 1.7-1 serving stack (CSV in, CSV out) on the model.tar.gz of the
training job. Serverless: billed per request, no idle instance. Removed by `make destroy`.
"""
from __future__ import annotations

import time

import boto3


def deploy(st, model_data_url: str, role_arn: str, memory_mb: int = 2048, max_concurrency: int = 2) -> str:
    from ml.train import xgb_image
    sm = boto3.client("sagemaker", region_name=st.region)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"{st.project_name}-forecast"
    model_name, config_name = f"{name}-model-{stamp}", f"{name}-config-{stamp}"
    tags = [{"Key": "Project", "Value": st.project_name}]
    sm.create_model(ModelName=model_name, ExecutionRoleArn=role_arn, Tags=tags,
                    PrimaryContainer={"Image": xgb_image(st.region), "ModelDataUrl": model_data_url})
    sm.create_endpoint_config(EndpointConfigName=config_name, Tags=tags, ProductionVariants=[{
        "VariantName": "AllTraffic", "ModelName": model_name,
        "ServerlessConfig": {"MemorySizeInMB": memory_mb, "MaxConcurrency": max_concurrency}}])
    try:
        sm.describe_endpoint(EndpointName=name)
        sm.update_endpoint(EndpointName=name, EndpointConfigName=config_name)
        print(f"Updating endpoint {name} ...")
    except sm.exceptions.ClientError:
        sm.create_endpoint(EndpointName=name, EndpointConfigName=config_name, Tags=tags)
        print(f"Creating serverless endpoint {name} ...")
    sm.get_waiter("endpoint_in_service").wait(EndpointName=name, WaiterConfig={"Delay": 30, "MaxAttempts": 40})
    print(f"Endpoint {name} InService")
    return name
