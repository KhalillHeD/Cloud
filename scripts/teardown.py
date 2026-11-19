#!/usr/bin/env python3
"""Remove every billable resource of the project (`make destroy`).

Order matters:
  1. SageMaker endpoints, endpoint configs, models and running jobs whose name starts
     with PROJECT_NAME (created by ml/train.py, outside CloudFormation, so the stack
     delete would not catch them; endpoints are the main cost risk).
  2. Empty every S3 bucket of the stack (CloudFormation cannot delete non-empty buckets).
  3. Delete the CloudFormation stack and wait.
  4. Delete the /aws/lambda/<stack>-* log groups Lambda created on its own.
  5. List anything still tagged Project=<PROJECT_NAME>.

Usage: python scripts/teardown.py [--dry-run] [--yes] [--keep-logs]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))

import boto3  # noqa: E402
from botocore.exceptions import ClientError  # noqa: E402
from traffic_common.config import Settings, load_dotenv  # noqa: E402


def log(msg: str) -> None:
    print(msg, flush=True)


def sagemaker_cleanup(sm, prefix: str, dry: bool) -> None:
    for ep in sm.get_paginator("list_endpoints").paginate(NameContains=prefix):
        for e in ep["Endpoints"]:
            log(f"  endpoint {e['EndpointName']} ({e['EndpointStatus']})")
            if not dry:
                sm.delete_endpoint(EndpointName=e["EndpointName"])
    for page in sm.get_paginator("list_endpoint_configs").paginate(NameContains=prefix):
        for c in page["EndpointConfigs"]:
            log(f"  endpoint config {c['EndpointConfigName']}")
            if not dry:
                sm.delete_endpoint_config(EndpointConfigName=c["EndpointConfigName"])
    for page in sm.get_paginator("list_models").paginate(NameContains=prefix):
        for m in page["Models"]:
            log(f"  model {m['ModelName']}")
            if not dry:
                sm.delete_model(ModelName=m["ModelName"])
    for page in sm.get_paginator("list_training_jobs").paginate(NameContains=prefix, StatusEquals="InProgress"):
        for j in page["TrainingJobSummaries"]:
            log(f"  stopping training job {j['TrainingJobName']}")
            if not dry:
                sm.stop_training_job(TrainingJobName=j["TrainingJobName"])
    for page in sm.get_paginator("list_transform_jobs").paginate(NameContains=prefix, StatusEquals="InProgress"):
        for j in page["TransformJobSummaries"]:
            log(f"  stopping transform job {j['TransformJobName']}")
            if not dry:
                sm.stop_transform_job(TransformJobName=j["TransformJobName"])


def empty_bucket(s3, name: str, dry: bool) -> None:
    try:
        pages = s3.get_paginator("list_object_versions").paginate(Bucket=name)
        n = 0
        for page in pages:
            objs = [{"Key": o["Key"], "VersionId": o["VersionId"]}
                    for o in page.get("Versions", []) + page.get("DeleteMarkers", [])]
            n += len(objs)
            if objs and not dry:
                s3.delete_objects(Bucket=name, Delete={"Objects": objs, "Quiet": True})
        log(f"  bucket {name}: {n} objects/versions {'would be ' if dry else ''}deleted")
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "NoSuchBucket":
            raise


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    st = Settings.from_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    ap.add_argument("--keep-logs", action="store_true")
    ap.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    args = ap.parse_args(argv)
    dry = args.dry_run

    session = boto3.Session(profile_name=args.profile or None, region_name=st.region)
    account = session.client("sts").get_caller_identity()["Account"]
    log(f"Teardown of stack '{st.stack_name}' and '{st.project_name}*' resources "
        f"in account {account}, {st.region}{' (dry run)' if dry else ''}")
    if not dry and not args.yes and input("Type the stack name to confirm: ").strip() != st.stack_name:
        log("Aborted.")
        return 1

    log("1. SageMaker")
    sagemaker_cleanup(session.client("sagemaker"), st.project_name, dry)

    cfn = session.client("cloudformation")
    try:
        resources = cfn.get_paginator("list_stack_resources").paginate(StackName=st.stack_name)
        buckets = [r["PhysicalResourceId"] for page in resources for r in page["StackResourceSummaries"]
                   if r["ResourceType"] == "AWS::S3::Bucket" and r.get("PhysicalResourceId")]
        stack_exists = True
    except ClientError as exc:
        if "does not exist" not in str(exc):
            raise
        buckets, stack_exists = [], False
        log(f"   stack {st.stack_name} does not exist")

    log("2. S3 buckets")
    s3 = session.client("s3")
    for b in buckets:
        empty_bucket(s3, b, dry)

    log("3. CloudFormation stack")
    if stack_exists and not dry:
        cfn.delete_stack(StackName=st.stack_name)
        log("   waiting for deletion (Kinesis/Firehose can take a few minutes)...")
        cfn.get_waiter("stack_delete_complete").wait(StackName=st.stack_name,
                                                     WaiterConfig={"Delay": 15, "MaxAttempts": 120})
        log("   stack deleted")

    if not args.keep_logs:
        log("4. Lambda log groups")
        logs = session.client("logs")
        for page in logs.get_paginator("describe_log_groups").paginate(
                logGroupNamePrefix=f"/aws/lambda/{st.stack_name}-"):
            for g in page["logGroups"]:
                log(f"   {g['logGroupName']}")
                if not dry:
                    logs.delete_log_group(logGroupName=g["logGroupName"])

    log("5. Leftovers tagged Project=" + st.project_name)
    try:
        tagging = session.client("resourcegroupstaggingapi")
        left = [r["ResourceARN"] for page in tagging.get_paginator("get_resources").paginate(
            TagFilters=[{"Key": "Project", "Values": [st.project_name]}]) for r in page["ResourceTagMappingList"]]
        for arn in left:
            log(f"   still present (may be eventual consistency): {arn}")
        log("   none" if not left else f"   {len(left)} resource(s) listed above")
    except ClientError as exc:
        log(f"   cannot list tagged resources ({exc.response['Error']['Code']}); check the console")
    log("Note: the 'aws-sam-cli-managed-default' stack (SAM artifact bucket) is shared and kept; "
        "it costs cents. Delete it in the CloudFormation console if you want a fully clean account.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
