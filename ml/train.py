#!/usr/bin/env python3
"""Block 8: train baselines + XGBoost, write docs/results.md, publish the model.

  --where local      train on this machine (minutes, free)
  --where sagemaker  SageMaker training job, ml.m5.large, XGBoost 1.7-1 container in script
                     mode running the same ml/train_core.py
  --where auto       sagemaker if .access_report.json says it is allowed, else local

After training (unless --no-publish): model.json goes to s3://<lake>/models/<run>/model.json
and app_state[model/active] points the forecast Lambda at it. With --endpoint (or when
the access report chose serverless-endpoint and training ran on SageMaker), a Serverless
Inference endpoint is created and the forecast Lambda calls it instead.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "common")]

from traffic_common.config import Settings, load_dotenv  # noqa: E402

from ml.evaluate import write_results  # noqa: E402

NPZ = ROOT / "data" / "processed" / "training.npz"
ART = ROOT / "ml" / "artifacts"
REPORT = ROOT / ".access_report.json"
XGB_IMAGE_ACCOUNT = {"us-east-1": "683313688378", "us-east-2": "257758044811", "us-west-2": "246618743249",
                     "eu-west-1": "141502667606"}


def decisions() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))["decisions"] if REPORT.exists() else {}


def xgb_image(region: str) -> str:
    return f"{XGB_IMAGE_ACCOUNT[region]}.dkr.ecr.{region}.amazonaws.com/sagemaker-xgboost:1.7-1"


def source_tarball() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(ROOT / "ml" / "sagemaker" / "sm_entry.py", arcname="sm_entry.py")
        for f in ("__init__.py", "features.py", "train_core.py", "evaluate.py"):
            tar.add(ROOT / "ml" / f, arcname=f"ml/{f}")
    return buf.getvalue()


def train_sagemaker(st: Settings, bucket: str, role: str, run: str, args) -> Path:
    import boto3
    s3 = boto3.client("s3", region_name=st.region)
    sm = boto3.client("sagemaker", region_name=st.region)
    prefix = f"ml/{run}"
    s3.upload_file(str(NPZ), bucket, f"{prefix}/input/training.npz")
    s3.put_object(Bucket=bucket, Key=f"{prefix}/code/sourcedir.tar.gz", Body=source_tarball())
    hp = {"sagemaker_program": "sm_entry.py",
          "sagemaker_submit_directory": f"s3://{bucket}/{prefix}/code/sourcedir.tar.gz",
          "sagemaker_region": st.region, "num_round": args.num_round, "max_train_rows": args.max_train_rows,
          "horizons": ",".join(map(str, st.forecast_horizons_min))}
    name = f"{st.project_name}-xgb-{run}"
    print(f"Starting SageMaker training job {name} (ml.m5.large) ...")
    sm.create_training_job(
        TrainingJobName=name, RoleArn=role,
        AlgorithmSpecification={"TrainingImage": xgb_image(st.region), "TrainingInputMode": "File"},
        HyperParameters={k: json.dumps(v) for k, v in hp.items()},
        InputDataConfig=[{"ChannelName": "train", "DataSource": {"S3DataSource": {
            "S3DataType": "S3Prefix", "S3Uri": f"s3://{bucket}/{prefix}/input/",
            "S3DataDistributionType": "FullyReplicated"}}}],
        OutputDataConfig={"S3OutputPath": f"s3://{bucket}/{prefix}/output/"},
        ResourceConfig={"InstanceType": "ml.m5.large", "InstanceCount": 1, "VolumeSizeInGB": 10},
        StoppingCondition={"MaxRuntimeInSeconds": 3600},
        Tags=[{"Key": "Project", "Value": st.project_name}])
    last = None
    while True:
        d = sm.describe_training_job(TrainingJobName=name)
        status = (d["TrainingJobStatus"], d.get("SecondaryStatus"))
        if status != last:
            print(f"  {status[0]} / {status[1]}", flush=True)
            last = status
        if d["TrainingJobStatus"] in ("Completed", "Failed", "Stopped"):
            break
        time.sleep(20)
    if d["TrainingJobStatus"] != "Completed":
        raise SystemExit(f"training job {d['TrainingJobStatus']}: {d.get('FailureReason')}")
    model_uri = d["ModelArtifacts"]["S3ModelArtifacts"]
    key = model_uri.split(f"s3://{bucket}/", 1)[1]
    tar_bytes = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    out = ART / run
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tar:
        for member in ("model.json", "metrics.json"):
            (out / member).write_bytes(tar.extractfile(member).read())
    (out / "sagemaker.json").write_text(json.dumps({"training_job": name, "model_data": model_uri}), encoding="utf-8")
    return out


def publish(st: Settings, out: Path, run: str, endpoint: str | None) -> None:
    from traffic_common import store
    key = f"models/{run}/model.json"
    store.client("s3").upload_file(str(out / "model.json"), store.lake_bucket(), key)
    store.client("s3").upload_file(str(out / "metrics.json"), store.lake_bucket(), f"models/{run}/metrics.json")
    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    summary = {h: round(per["xgboost"]["mae"], 3) for h, per in m["horizons"].items()}
    store.put_state("model", "active", {"mode": "sagemaker" if endpoint else "lambda", "s3_key": key,
                                        "endpoint": endpoint, "run": run, "trained_on": m["trained_on"],
                                        "mae": summary, "horizons_min": m["horizons_min"]})
    print(f"Published model {run} ({'endpoint ' + endpoint if endpoint else 'in-Lambda evaluation'}).")


def main(argv=None) -> int:
    load_dotenv(ROOT / ".env")
    st = Settings.from_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--where", choices=["auto", "local", "sagemaker"], default="auto")
    ap.add_argument("--num-round", type=int, default=300)
    ap.add_argument("--max-train-rows", type=int, default=1_500_000)
    ap.add_argument("--endpoint", action="store_true", help="deploy a serverless endpoint (SageMaker runs only)")
    ap.add_argument("--no-publish", action="store_true", help="train and evaluate only")
    args = ap.parse_args(argv)
    if not NPZ.exists():
        print("data/processed/training.npz missing: run `python tasks.py eda` first.")
        return 1
    d = decisions()
    where = args.where
    if where == "auto":
        where = "sagemaker" if d.get("training", "").startswith("sagemaker") else "local"
    run = time.strftime("%Y%m%d-%H%M%S")
    if where == "local":
        from ml.train_core import train
        out = ART / run
        train(NPZ, out, st.forecast_horizons_min, args.num_round, args.max_train_rows,
              trained_on="local machine")
    else:
        from traffic_common import store
        role = d.get("lab_role_arn") or st.lab_role_arn
        if not role:
            raise SystemExit("No LabRole ARN: run `python tasks.py check` first.")
        out = train_sagemaker(st, store.lake_bucket(), role, run, args)
    (ART / "latest").mkdir(parents=True, exist_ok=True)
    for f in ("model.json", "metrics.json"):
        (ART / "latest" / f).write_bytes((out / f).read_bytes())
    print(write_results(out / "metrics.json", ROOT / "docs" / "results.md"))
    if args.no_publish:
        return 0
    endpoint = None
    want_endpoint = args.endpoint or d.get("inference") == "serverless-endpoint"
    if where == "sagemaker" and want_endpoint:
        from ml.deploy_endpoint import deploy
        endpoint = deploy(st, json.loads((out / "sagemaker.json").read_text())["model_data"],
                          d.get("lab_role_arn") or st.lab_role_arn)
    publish(st, out, run, endpoint)
    return 0


if __name__ == "__main__":
    sys.exit(main())
