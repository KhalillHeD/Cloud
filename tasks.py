#!/usr/bin/env python3
"""Cross-platform task runner.

`make <target>` simply calls `python tasks.py <target>`. On Windows without GNU make,
run `python tasks.py <target>` directly; extra arguments go after the target, e.g.
`python tasks.py check --probe-write` (or `make check ARGS=--probe-write`).

Typical first run:   setup -> check --probe-write -> deploy -> demo
Without AWS:         setup -> download -> eda -> train -> local
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "common")]

from traffic_common.config import Settings, load_dotenv  # noqa: E402

VENV = ROOT / ".venv"
VENV_PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
PY = str(VENV_PY) if VENV_PY.exists() else sys.executable
TEMPLATE = ROOT / "infra" / "template.yaml"
REPORT = ROOT / ".access_report.json"
OUTPUTS = ROOT / ".stack_outputs.json"
PARAMS = ROOT / ".deploy_params.json"
MODEL = ROOT / "ml" / "artifacts" / "latest" / "model.json"
PROCESSED = ROOT / "data" / "processed" / "sensor_meta.json"

TASKS: dict[str, tuple] = {}


def task(help_text: str):
    def deco(fn):
        TASKS[fn.__name__.strip("_").replace("_", "-")] = (fn, help_text)
        return fn
    return deco


def run(cmd: list[str], env: dict | None = None, **kw) -> None:
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT, env={**os.environ, **(env or {})}, **kw)


def py(script: str, *args, env: dict | None = None) -> None:
    run([PY, script, *args], env=env)


# ----------------------------------------------------------------------------- local work

@task("list targets")
def _help(_args):
    print(__doc__)
    for name, (_, h) in TASKS.items():
        print(f"  {name:<10} {h}")


@task("create .venv, install dev + data/ML dependencies, create .env")
def setup(_args):
    global PY
    if not VENV_PY.exists():
        run([sys.executable, "-m", "venv", VENV])
        PY = str(VENV_PY)
    run([PY, "-m", "pip", "install", "--upgrade", "pip"])
    run([PY, "-m", "pip", "install", "-r", "requirements-dev.txt", "-r", "requirements-ml.txt"])
    if not (ROOT / ".env").exists():
        shutil.copy(ROOT / ".env.example", ROOT / ".env")
        print("Created .env from .env.example: review it (ALERT_EMAIL at least).")


@task("probe AWS service access (add --probe-write for create/delete tests)")
def check(args):
    py("scripts/check_access.py", *args)


@task("download METR-LA into data/")
def download(args):
    py("scripts/download_data.py", *args)


@task("precompute free-flow, expected times, replay files; build + run the EDA notebook")
def eda(args):
    if not (ROOT / "data" / "metr-la.h5").exists():
        download([])
    py("scripts/precompute_freeflow.py", *args)
    py("scripts/build_eda_notebook.py")


@task("train baselines + XGBoost (--where local|sagemaker|auto), write docs/results.md")
def train(args):
    if not (ROOT / "data" / "processed" / "training.npz").exists():
        eda([])
    py("ml/train.py", *args, env=_aws_env())


@task("run the whole pipeline locally on an AWS mock + dashboard at http://localhost:8080")
def local(args):
    if not PROCESSED.exists():
        eda([])
    if not MODEL.exists():
        py("ml/train.py", "--where", "local", "--no-publish")
    py("scripts/local_run.py", *args)


@task("run the pytest suite")
def test(args):
    run([PY, "-m", "pytest", *args])


@task("ruff lint")
def lint(args):
    run([PY, "-m", "ruff", "check", ".", *args])


@task("cfn-lint (and sam validate, if SAM CLI is installed) on infra/template.yaml")
def validate(_args):
    run([PY, "-c", "import sys, cfnlint.runner; sys.exit(cfnlint.runner.main())", TEMPLATE])
    sam = _sam(required=False)
    if sam:
        run([sam, "validate", "--lint", "--template-file", TEMPLATE, "--region", Settings.from_env().region])
    else:
        print("SAM CLI not installed: skipped `sam validate` (cfn-lint passed).")


@task("export the allow-listed Athena queries to glue/athena/*.sql")
def sql(_args):
    from traffic_common import history_sql
    out = ROOT / "glue" / "athena"
    out.mkdir(parents=True, exist_ok=True)
    for name, spec in history_sql.TEMPLATES.items():
        (out / f"{name}.sql").write_text(f"-- {spec['description']}\n-- defaults: "
                                         f"{ {k: v[3] for k, v in spec['params'].items()} }\n"
                                         + history_sql.render(name) + ";\n", encoding="utf-8")
    print(f"wrote {len(history_sql.TEMPLATES)} queries to {out}")


# ----------------------------------------------------------------------------- AWS

def _sam(required: bool = True) -> str | None:
    sam = shutil.which("sam") or shutil.which("sam.cmd")
    if not sam and required:
        sys.exit("AWS SAM CLI not found: `winget install Amazon.SAM-CLI` (see README > Prerequisites).")
    return sam


def _decisions() -> dict:
    if not REPORT.exists():
        print("No .access_report.json yet: running the access check first.")
        check([])
    return json.loads(REPORT.read_text(encoding="utf-8"))["decisions"]


def _use_container() -> bool:
    """sam build needs a python3.11 interpreter for python3.11 Lambdas; otherwise build in Docker."""
    mode = os.environ.get("SAM_USE_CONTAINER", "auto").lower()
    if mode in ("1", "true", "yes"):
        return True
    if mode in ("0", "false", "no"):
        return False
    if sys.version_info[:2] == (3, 11) or shutil.which("python3.11"):
        return False
    try:
        out = subprocess.run(["py", "-3.11", "-c", "print(1)"], capture_output=True, text=True, timeout=20)
        return out.returncode != 0
    except (OSError, subprocess.TimeoutExpired):
        return True


def _outputs() -> dict:
    if not OUTPUTS.exists():
        sys.exit("No stack outputs: run `python tasks.py deploy` first.")
    return json.loads(OUTPUTS.read_text(encoding="utf-8"))


def _aws_env() -> dict:
    """Environment for scripts that talk to the deployed stack."""
    env = {}
    if OUTPUTS.exists():
        out = json.loads(OUTPUTS.read_text(encoding="utf-8"))
        env["LAKE_BUCKET"] = out["DataLakeBucketName"]
        env["ALERT_TOPIC_ARN"] = out["AlertTopicArn"]
        env["ATHENA_WORKGROUP"] = out["AthenaWorkGroupName"]
        env["GLUE_DATABASE"] = out["GlueDatabaseName"]
    return env


def _base_params(st: Settings, d: dict) -> dict:
    llm = d.get("llm", {})
    secret = os.environ.get("LLM_API_KEY_SECRET_NAME", "")
    if st.llm_provider != "bedrock":
        provider = st.llm_provider
    elif llm.get("provider") == "bedrock":
        provider = "bedrock"
    else:
        provider = "anthropic" if secret else "none"
        print(f"Bedrock is not usable in this account -> LLM provider '{provider}'"
              + ("" if secret else " (set LLM_API_KEY_SECRET_NAME to use an external LLM)"))
    return {
        "ProjectName": st.project_name,
        "LabRoleArn": st.lab_role_arn or d["lab_role_arn"],
        "NSensors": st.n_sensors,
        "TickSeconds": st.tick_seconds,
        "ReplayMode": st.replay_mode,
        "ReplayStart": st.replay_start,
        "SyntheticIncidentRate": st.synthetic_incident_rate,
        "ProducerState": "ENABLED",
        "SchedulerType": d.get("scheduler", "rule"),
        "AlertEmail": st.alert_email or "",
        "LlmProvider": provider,
        "BedrockModelId": st.bedrock_model_id or llm.get("model_id") or "",
        "LlmApiKeySecretName": secret,
        "UseCloudFront": "true" if os.environ.get("USE_CLOUDFRONT", "false").lower() == "true"
                         and d.get("frontend_cdn") == "cloudfront" else "false",
    }


def sam_deploy(overrides: dict | None = None, build: bool = True) -> dict:
    st = Settings.from_env()
    d = _decisions()
    if not d.get("lab_role_arn"):
        sys.exit("The access report has no LabRole ARN: fix credentials and rerun `python tasks.py check`.")
    params = {**_base_params(st, d), **(json.loads(PARAMS.read_text()) if PARAMS.exists() and overrides else {}),
              **(overrides or {})}
    sam = _sam()
    if build or not (ROOT / ".aws-sam" / "build" / "template.yaml").exists():
        cmd = [sam, "build", "--template-file", TEMPLATE, "--build-dir", ".aws-sam/build", "--cached"]
        if _use_container():
            print("No local Python 3.11: building the Lambdas in Docker (Docker Desktop must be running).")
            cmd.append("--use-container")
        run(cmd)
    run([sam, "deploy", "--template-file", ".aws-sam/build/template.yaml",
         "--stack-name", st.stack_name, "--region", st.region, "--resolve-s3",
         "--capabilities", "CAPABILITY_AUTO_EXPAND", "--no-confirm-changeset", "--no-fail-on-empty-changeset",
         "--tags", f"Project={st.project_name}",
         "--parameter-overrides", *[f"{k}={v}" for k, v in params.items() if str(v) != ""]])
    PARAMS.write_text(json.dumps(params, indent=2), encoding="utf-8")
    import boto3
    cfn = boto3.client("cloudformation", region_name=st.region)
    outs = {o["OutputKey"]: o["OutputValue"]
            for o in cfn.describe_stacks(StackName=st.stack_name)["Stacks"][0].get("Outputs", [])}
    OUTPUTS.write_text(json.dumps(outs, indent=2), encoding="utf-8")
    return outs


def upload_dashboard(outs: dict, tick_seconds: int) -> None:
    import boto3
    s3 = boto3.client("s3", region_name=Settings.from_env().region)
    bucket = outs["DashboardBucketName"]
    types = {".html": "text/html", ".js": "application/javascript", ".css": "text/css"}
    for f in (ROOT / "dashboard").iterdir():
        if f.suffix in types and f.name != "config.js":
            s3.upload_file(str(f), bucket, f.name, ExtraArgs={"ContentType": types[f.suffix], "CacheControl": "no-cache"})
    cfg = {"apiBase": outs["HttpApiUrl"], "mode": "aws", "tickSeconds": tick_seconds}
    s3.put_object(Bucket=bucket, Key="config.js", Body=f"window.APP_CONFIG = {json.dumps(cfg)};".encode(),
                  ContentType="application/javascript", CacheControl="no-cache")
    (ROOT / "dashboard" / "config.local.js").write_text(f"window.APP_CONFIG = {json.dumps(cfg)};\n", encoding="utf-8")
    print(f"Dashboard uploaded to s3://{bucket}")


def sync_named_queries(outs: dict) -> None:
    import boto3
    from traffic_common import history_sql
    ath = boto3.client("athena", region_name=Settings.from_env().region)
    wg, db = outs["AthenaWorkGroupName"], outs["GlueDatabaseName"]
    ids = ath.list_named_queries(WorkGroup=wg).get("NamedQueryIds", [])
    if ids:
        for q in ath.batch_get_named_query(NamedQueryIds=ids)["NamedQueries"]:
            if q["Name"] in history_sql.TEMPLATES:
                ath.delete_named_query(NamedQueryId=q["NamedQueryId"])
    for name, spec in history_sql.TEMPLATES.items():
        ath.create_named_query(Name=name, Database=db, WorkGroup=wg, Description=spec["description"][:1024],
                               QueryString=history_sql.render(name))
    print(f"Athena: {len(history_sql.TEMPLATES)} named queries in workgroup {wg}")


@task("deploy everything: stack, data seed, model, dashboard (one command)")
def deploy(args):
    st = Settings.from_env()
    if not PROCESSED.exists():
        eda([])
    outs = sam_deploy()
    env = _aws_env()
    py("scripts/seed_dynamo.py", "meta", "replay", "lake", env=env)
    import boto3
    ddb = boto3.resource("dynamodb", region_name=st.region)
    has_model = "Item" in ddb.Table(f"{st.project_name}-app_state").get_item(Key={"pk": "model", "sk": "active"})
    if not has_model:
        if MODEL.exists() and "--retrain" not in args:
            py("scripts/seed_dynamo.py", "model", env=env)
        else:
            py("ml/train.py", "--where", "auto", env=env)
    py("scripts/seed_dynamo.py", "reset", env=env)
    upload_dashboard(outs, st.tick_seconds)
    sync_named_queries(outs)
    sql([])
    print("\nDeployed.")
    print(f"  Dashboard : {outs['DashboardUrl']}")
    print(f"  API       : {outs['HttpApiUrl']}")
    if st.alert_email:
        print(f"  Alerts    : confirm the SNS subscription email sent to {st.alert_email}")
    print("Next: `python tasks.py demo` (fast replay, 1 tick / 10 s).")


@task("demo mode: 1 tick every 10 s from REPLAY_START (Monday 06:00) + open the dashboard")
def demo(args):
    tick = int(args[0]) if args else 10
    outs = sam_deploy({"TickSeconds": tick, "ReplayMode": "cursor", "ProducerState": "ENABLED"}, build=False)
    py("scripts/seed_dynamo.py", "reset", env={**_aws_env(), "REPLAY_MODE": "cursor", "TICK_SECONDS": str(tick)})
    upload_dashboard(outs, tick)
    print(f"\nDemo running: one 5-minute slice every {tick} s, starting {Settings.from_env().replay_start}.")
    print(f"  Dashboard: {outs['DashboardUrl']}")
    print("  Stop with `python tasks.py pause` (or `python tasks.py destroy` when finished).")
    webbrowser.open(outs["DashboardUrl"])


@task("pause the replay and forecasts (schedules disabled); `deploy` or `demo` resumes")
def pause(_args):
    sam_deploy({"ProducerState": "DISABLED"}, build=False)
    print("Paused: no producer/forecast invocations until the next deploy/demo.")


@task("run the Glue crawler + ETL job now and wait")
def etl(_args):
    import boto3
    outs, st = _outputs(), Settings.from_env()
    glue = boto3.client("glue", region_name=st.region)
    crawler, job = outs["RawCrawlerName"], outs["EtlJobName"]
    try:
        glue.start_crawler(Name=crawler)
        print(f"crawler {crawler} started")
    except glue.exceptions.CrawlerRunningException:
        print(f"crawler {crawler} already running")
    run_id = glue.start_job_run(JobName=job)["JobRunId"]
    print(f"job {job} run {run_id} started (Glue jobs take ~2-3 min to start) ...")
    while True:
        r = glue.get_job_run(JobName=job, RunId=run_id)["JobRun"]
        if r["JobRunState"] in ("SUCCEEDED", "FAILED", "STOPPED", "TIMEOUT", "ERROR"):
            print(f"  {r['JobRunState']} {r.get('ErrorMessage', '')}")
            break
        time.sleep(20)


def _invoke_agent(payload: dict) -> dict:
    import boto3
    lam = boto3.client("lambda", region_name=Settings.from_env().region)
    r = lam.invoke(FunctionName=_outputs()["AgentFunctionName"], Payload=json.dumps(payload).encode())
    return json.loads(r["Payload"].read())


@task("generate the daily report now (S3 reports/ + email)")
def report(_args):
    print(json.dumps(_invoke_agent({"mode": "daily_report"}), indent=2))


@task('ask the deployed agent a question, e.g. ask "Which roads will be jammed in 30 minutes?"')
def ask(args):
    q = " ".join(args) or "Which roads will be congested in 30 minutes, and why?"
    r = _invoke_agent({"requestContext": {}, "body": json.dumps({"message": q})})
    body = json.loads(r.get("body", "{}"))
    print(body.get("answer") or body)
    for c in body.get("tool_calls", []):
        print(f"  tool call: {c['name']}({json.dumps(c['args'])})")


@task("show the deployed URLs and the active model")
def status(_args):
    outs = _outputs()
    for k in ("DashboardUrl", "HttpApiUrl", "DataLakeBucketName", "AlertTopicArn"):
        print(f"  {k:<20} {outs.get(k)}")
    if PARAMS.exists():
        p = json.loads(PARAMS.read_text())
        print(f"  tick={p['TickSeconds']}s mode={p['ReplayMode']} producer={p['ProducerState']} llm={p['LlmProvider']}")


@task("delete every billable resource (stack, SageMaker endpoints, buckets, logs)")
def destroy(args):
    py("scripts/teardown.py", *args)
    for f in (OUTPUTS, PARAMS):
        f.unlink(missing_ok=True)


@task("deploy, then start the demo (same as `deploy` + `demo`)")
def up(args):
    deploy(args)
    demo([])


def main() -> None:
    load_dotenv(ROOT / ".env")
    name, *args = sys.argv[1:] or ["help"]
    if name not in TASKS:
        print(f"Unknown target '{name}'.")
        _help([])
        sys.exit(1)
    try:
        TASKS[name][0](args)
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)


if __name__ == "__main__":
    main()
