#!/usr/bin/env python3
"""Phase 0 - probe which AWS services this account can actually use.

Read-only by default: it lists resources in every service the 10 blocks rely on.
With --probe-write it also creates tiny throwaway resources (S3 bucket, DynamoDB table,
SNS topic, Kinesis stream, Lambda using LabRole, EventBridge rule + Scheduler schedule)
and deletes them straight away. That is the only reliable way to know whether
iam:PassRole on LabRole works for each service in AWS Academy Learner Lab.
Cost of a full probe run: well under $0.01.

Results are printed as a table and saved to .access_report.json; `tasks.py deploy`
reads the "decisions" section from it (scheduler type, LLM provider, inference mode).

Usage:
    python scripts/check_access.py [--region us-east-1] [--profile NAME] [--probe-write]
Exit codes: 0 all critical checks ok, 1 a critical check failed, 2 no usable credentials.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import uuid
import zipfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))

from traffic_common.config import Settings, load_dotenv  # noqa: E402

try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import BotoCoreError, ClientError, EndpointConnectionError, NoCredentialsError
except ImportError:  # pragma: no cover
    print("boto3 is not installed: run `python tasks.py setup` first.")
    sys.exit(2)

OK, DENIED, ERROR, SKIPPED = "ok", "denied", "error", "skipped"
DENIED_CODES = {
    "AccessDenied", "AccessDeniedException", "UnauthorizedOperation", "AuthorizationError",
    "UnauthorizedException", "NotAuthorized", "Forbidden", "ForbiddenException",
}
CREDENTIAL_CODES = {"ExpiredToken", "ExpiredTokenException", "InvalidClientTokenId",
                    "UnrecognizedClientException", "SignatureDoesNotMatch"}
BOTO_CFG = Config(retries={"max_attempts": 2, "mode": "standard"}, connect_timeout=5, read_timeout=30)


class Skip(Exception):
    """Raised by a check that cannot run (missing prerequisite); not a failure."""


@dataclass
class Check:
    key: str
    block: str
    critical: bool
    write: bool
    fn: Callable[[Any, dict], str]


@dataclass
class Result:
    key: str
    block: str
    critical: bool
    status: str
    detail: str
    seconds: float = 0.0


@dataclass
class Report:
    generated_at: str
    region: str
    identity: dict = field(default_factory=dict)
    results: list[Result] = field(default_factory=list)
    decisions: dict = field(default_factory=dict)


CHECKS: list[Check] = []


def check(key: str, block: str, critical: bool = True, write: bool = False):
    def deco(fn):
        CHECKS.append(Check(key, block, critical, write, fn))
        return fn
    return deco


def client(session, name: str):
    return session.client(name, config=BOTO_CFG)


def classify(exc: BaseException) -> tuple[str, str]:
    """Map an exception to (status, short detail)."""
    if isinstance(exc, Skip):
        return SKIPPED, str(exc)
    if isinstance(exc, ClientError):
        err = exc.response.get("Error", {})
        code, msg = err.get("Code", "?"), err.get("Message", "")
        if msg.lstrip().startswith("<"):  # HTML error page (5xx from a proxy or mock)
            msg = f"HTTP {exc.response.get('ResponseMetadata', {}).get('HTTPStatusCode', '?')} (non-JSON body)"
        msg = " ".join(msg.split())
        denied = code in DENIED_CODES or "AccessDenied" in code or "not authorized" in msg.lower()
        return (DENIED if denied else ERROR), f"{code}: {msg[:160]}"
    if isinstance(exc, NoCredentialsError):
        return ERROR, "no AWS credentials found"
    if isinstance(exc, EndpointConnectionError):
        return ERROR, f"endpoint unreachable (service not offered in this region?): {exc}"
    if isinstance(exc, BotoCoreError):
        return ERROR, f"{type(exc).__name__}: {exc}"
    return ERROR, f"{type(exc).__name__}: {exc}"


def run_checks(session, checks: list[Check], ctx: dict, probe_write: bool) -> list[Result]:
    results = []
    for c in checks:
        if c.write and not probe_write:
            results.append(Result(c.key, c.block, c.critical, SKIPPED, "write probe: rerun with --probe-write"))
            continue
        start = time.monotonic()
        try:
            detail = c.fn(session, ctx) or ""
            status = OK
        except Exception as exc:  # noqa: BLE001 - every failure must be reported, never crash
            status, detail = classify(exc)
        results.append(Result(c.key, c.block, c.critical, status, detail, round(time.monotonic() - start, 1)))
        ctx.setdefault("status", {})[c.key] = status
    return results


# ----------------------------------------------------------------------------- read checks

@check("iam.labrole", "infra (all blocks)")
def _labrole(s, ctx):
    try:
        arn = client(s, "iam").get_role(RoleName="LabRole")["Role"]["Arn"]
        ctx["lab_role_arn"] = arn
        return arn
    except ClientError as exc:
        if classify(exc)[0] == DENIED:  # Learner Lab may hide IAM reads; the role still exists
            ctx["lab_role_arn"] = ctx["settings"].resolve_lab_role_arn(ctx["account"])
            return f"iam:GetRole denied, assuming {ctx['lab_role_arn']} (verified by --probe-write)"
        raise


@check("cloudformation.list", "infra (SAM deploy)")
def _cfn(s, ctx):
    stacks = client(s, "cloudformation").list_stacks(StackStatusFilter=["CREATE_COMPLETE", "UPDATE_COMPLETE"])
    return f"{len(stacks['StackSummaries'])} active stacks"


@check("s3.list", "6/7 - S3 data lake, 10 - site")
def _s3(s, ctx):
    return f"{len(client(s, 's3').list_buckets()['Buckets'])} buckets"


@check("lambda.list", "2/4/8/9/10 - Lambda")
def _lambda(s, ctx):
    return f"{len(client(s, 'lambda').list_functions(MaxItems=50)['Functions'])} functions (first page)"


@check("events.list", "2 - EventBridge rules")
def _events(s, ctx):
    return f"{len(client(s, 'events').list_rules(Limit=50)['Rules'])} rules"


@check("scheduler.list", "2 - EventBridge Scheduler", critical=False)
def _scheduler(s, ctx):
    return f"{len(client(s, 'scheduler').list_schedules(MaxResults=50)['Schedules'])} schedules"


@check("kinesis.list", "3 - Kinesis Data Streams")
def _kinesis(s, ctx):
    return f"{len(client(s, 'kinesis').list_streams(Limit=50)['StreamNames'])} streams"


@check("dynamodb.list", "5 - DynamoDB")
def _ddb(s, ctx):
    return f"{len(client(s, 'dynamodb').list_tables(Limit=50)['TableNames'])} tables"


@check("firehose.list", "6 - Data Firehose")
def _firehose(s, ctx):
    return f"{len(client(s, 'firehose').list_delivery_streams(Limit=50)['DeliveryStreamNames'])} delivery streams"


@check("glue.list", "7 - Glue")
def _glue(s, ctx):
    g = client(s, "glue")
    dbs = g.get_databases(MaxResults=50)["DatabaseList"]
    crawlers = g.list_crawlers(MaxResults=50)["CrawlerNames"]
    return f"{len(dbs)} databases, {len(crawlers)} crawlers"


@check("athena.list", "7 - Athena")
def _athena(s, ctx):
    wgs = client(s, "athena").list_work_groups(MaxResults=50)["WorkGroups"]
    return "workgroups: " + ", ".join(w["Name"] for w in wgs)


@check("sagemaker.training", "8 - SageMaker training")
def _sm_train(s, ctx):
    jobs = client(s, "sagemaker").list_training_jobs(MaxResults=10)["TrainingJobSummaries"]
    return f"{len(jobs)} recent training jobs"


@check("sagemaker.endpoints", "8 - SageMaker endpoints", critical=False)
def _sm_ep(s, ctx):
    eps = client(s, "sagemaker").list_endpoints(MaxResults=50)["Endpoints"]
    return f"{len(eps)} endpoints"


@check("sagemaker.quotas", "8 - SageMaker quotas", critical=False)
def _sm_quotas(s, ctx):
    wanted = {
        "ml.m5.large for training job usage": "train_m5_large",
        "ml.m5.large for transform job usage": "transform_m5_large",
        "Maximum total concurrency that can be allocated across all serverless endpoints": "serverless_concurrency",
        "Maximum number of serverless endpoints": "serverless_endpoints",
    }
    sq = client(s, "service-quotas")
    found: dict[str, float] = {}
    for op in ("list_service_quotas", "list_aws_default_service_quotas"):
        for page in sq.get_paginator(op).paginate(ServiceCode="sagemaker"):
            for q in page["Quotas"]:
                key = wanted.get(q["QuotaName"])
                if key and key not in found:
                    found[key] = q["Value"]
        if len(found) == len(wanted):
            break
    ctx["sagemaker_quotas"] = found
    return ", ".join(f"{k}={v:g}" for k, v in sorted(found.items())) or "no matching quotas returned"


def _rank_models(ids: list[str]) -> list[str]:
    """Cheap, capable chat models first: Claude Haiku, then other Claude, then Amazon Nova."""
    def score(mid: str) -> tuple:
        m = mid.lower()
        family = 0 if ("anthropic" in m and "haiku" in m) else 1 if "anthropic" in m else 2 if "nova" in m else 9
        regional = 0 if m.startswith(("us.", "global.")) else 1   # newer models need an inference profile
        return (family, regional, [-ord(ch) for ch in m])         # reverse-lexical ~ newest version first
    return sorted({i for i in ids if any(k in i.lower() for k in ("anthropic", "nova"))}, key=score)


@check("bedrock.models", "9 - Bedrock", critical=False)
def _bedrock_models(s, ctx):
    b = client(s, "bedrock")
    models = b.list_foundation_models(byOutputModality="TEXT")["modelSummaries"]
    on_demand = [m["modelId"] for m in models if "ON_DEMAND" in m.get("inferenceTypesSupported", [])]
    profiles: list[str] = []
    try:
        profiles = [p["inferenceProfileId"]
                    for p in b.list_inference_profiles(maxResults=100)["inferenceProfileSummaries"]]
    except ClientError:
        pass
    ctx["bedrock_candidates"] = _rank_models(profiles + on_demand)
    top = ", ".join(ctx["bedrock_candidates"][:3]) or "none"
    return f"{len(models)} text models, {len(profiles)} inference profiles; best candidates: {top}"


@check("bedrock.invoke", "9 - Bedrock", critical=False)
def _bedrock_invoke(s, ctx):
    forced = ctx["settings"].bedrock_model_id
    candidates = [forced] if forced else ctx.get("bedrock_candidates", [])[:4]
    if not candidates:
        raise Skip("no candidate model (listing failed or empty); set BEDROCK_MODEL_ID to force one")
    rt = client(s, "bedrock-runtime")
    tried = []
    for mid in candidates:
        try:
            rt.converse(modelId=mid, messages=[{"role": "user", "content": [{"text": "Reply with: OK"}]}],
                        inferenceConfig={"maxTokens": 5})
            ctx["bedrock_model_id"] = mid
            return f"invoked {mid}" + (f" (after {len(tried)} failures)" if tried else "")
        except ClientError as exc:
            tried.append(f"{mid} -> {exc.response['Error']['Code']}")
    raise ClientError({"Error": {"Code": "AccessDeniedException" if all("AccessDenied" in t for t in tried)
                                 else "InvokeFailed", "Message": "; ".join(tried)}}, "Converse")


@check("sns.list", "9 - SNS alerts")
def _sns(s, ctx):
    return f"{len(client(s, 'sns').list_topics()['Topics'])} topics"


@check("secretsmanager.list", "9 - LLM fallback key", critical=False)
def _secrets(s, ctx):
    return f"{len(client(s, 'secretsmanager').list_secrets(MaxResults=20)['SecretList'])} secrets"


@check("apigatewayv2.list", "10 - API Gateway (HTTP API)")
def _apigw(s, ctx):
    return f"{len(client(s, 'apigatewayv2').get_apis(MaxResults='50')['Items'])} HTTP APIs"


@check("cloudfront.list", "10 - CloudFront", critical=False)
def _cloudfront(s, ctx):
    dl = client(s, "cloudfront").list_distributions(MaxItems="10")["DistributionList"]
    return f"{dl.get('Quantity', 0)} distributions"


@check("logs.list", "all - CloudWatch Logs")
def _logs(s, ctx):
    return f"{len(client(s, 'logs').describe_log_groups(limit=10)['logGroups'])} log groups (first page)"


# ----------------------------------------------------------------------------- write probes

def _probe_name(ctx, kind: str) -> str:
    return f"{ctx['settings'].project_name}-probe-{kind}-{ctx['run_id']}"


@check("s3.create", "6/7 - S3", write=True)
def _p_s3(s, ctx):
    s3 = client(s, "s3")
    name = _probe_name(ctx, "s3")
    kwargs = {} if s.region_name == "us-east-1" else {
        "CreateBucketConfiguration": {"LocationConstraint": s.region_name}}
    s3.create_bucket(Bucket=name, **kwargs)
    s3.delete_bucket(Bucket=name)
    return "created + deleted bucket"


@check("dynamodb.create", "5 - DynamoDB", write=True)
def _p_ddb(s, ctx):
    d = client(s, "dynamodb")
    name = _probe_name(ctx, "ddb")
    d.create_table(TableName=name, BillingMode="PAY_PER_REQUEST",
                   AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "S"}],
                   KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}])
    try:
        d.get_waiter("table_exists").wait(TableName=name, WaiterConfig={"Delay": 2, "MaxAttempts": 30})
    finally:
        d.delete_table(TableName=name)
    return "created on-demand table + deleted"


@check("sns.create", "9 - SNS", write=True)
def _p_sns(s, ctx):
    sns = client(s, "sns")
    arn = sns.create_topic(Name=_probe_name(ctx, "sns"))["TopicArn"]
    sns.delete_topic(TopicArn=arn)
    return "created + deleted topic"


@check("kinesis.create", "3 - Kinesis", write=True)
def _p_kinesis(s, ctx):
    k = client(s, "kinesis")
    name = _probe_name(ctx, "kds")
    k.create_stream(StreamName=name, StreamModeDetails={"StreamMode": "ON_DEMAND"})
    try:
        k.get_waiter("stream_exists").wait(StreamName=name, WaiterConfig={"Delay": 3, "MaxAttempts": 40})
    finally:
        k.delete_stream(StreamName=name, EnforceConsumerDeletion=True)
    return "created ON_DEMAND stream + deleted"


@check("lambda.create_with_labrole", "2/4/8/9/10 - Lambda + LabRole", write=True)
def _p_lambda(s, ctx):
    lam = client(s, "lambda")
    name = _probe_name(ctx, "fn")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("app.py", "def handler(event, context):\n    return {'ok': True}\n")
    arn = lam.create_function(FunctionName=name, Runtime="python3.11", Role=ctx["lab_role_arn"],
                              Handler="app.handler", Code={"ZipFile": buf.getvalue()},
                              Timeout=10, MemorySize=128)["FunctionArn"]
    ctx["probe_fn"] = (name, arn)
    lam.get_waiter("function_active_v2").wait(FunctionName=name, WaiterConfig={"Delay": 2, "MaxAttempts": 30})
    resp = lam.invoke(FunctionName=name)
    if resp.get("StatusCode") != 200 or "FunctionError" in resp:
        raise RuntimeError(f"invoke failed: {resp.get('FunctionError')} {resp['Payload'].read()[:200]!r}")
    return "created python3.11 function with LabRole, invoked it"


@check("events.create_rule", "2 - EventBridge rule -> Lambda", write=True)
def _p_rule(s, ctx):
    if "probe_fn" not in ctx:
        raise Skip("needs the Lambda probe")
    ev = client(s, "events")
    name = _probe_name(ctx, "rule")
    ev.put_rule(Name=name, ScheduleExpression="rate(1 day)", State="DISABLED")
    try:
        ev.put_targets(Rule=name, Targets=[{"Id": "probe", "Arn": ctx["probe_fn"][1]}])
        ev.remove_targets(Rule=name, Ids=["probe"])
    finally:
        ev.delete_rule(Name=name)
    return "rule + Lambda target created and deleted (no role needed)"


@check("scheduler.create", "2 - EventBridge Scheduler (LabRole)", critical=False, write=True)
def _p_scheduler(s, ctx):
    if "probe_fn" not in ctx:
        raise Skip("needs the Lambda probe")
    sch = client(s, "scheduler")
    name = _probe_name(ctx, "sch")
    sch.create_schedule(Name=name, ScheduleExpression="rate(1 day)", State="DISABLED",
                        FlexibleTimeWindow={"Mode": "OFF"},
                        Target={"Arn": ctx["probe_fn"][1], "RoleArn": ctx["lab_role_arn"]})
    sch.delete_schedule(Name=name)
    return "schedule with LabRole created + deleted"


@check("lambda.cleanup", "cleanup", critical=False, write=True)
def _p_cleanup(s, ctx):
    if "probe_fn" not in ctx:
        raise Skip("nothing to clean")
    client(s, "lambda").delete_function(FunctionName=ctx["probe_fn"][0])
    try:
        client(s, "logs").delete_log_group(logGroupName=f"/aws/lambda/{ctx['probe_fn'][0]}")
    except ClientError:
        pass  # no log group if the function never ran
    return "probe function (and its log group) deleted"


# ----------------------------------------------------------------------------- decisions

def decide(results: list[Result], ctx: dict) -> dict:
    st = {r.key: r.status for r in results}
    quotas = ctx.get("sagemaker_quotas", {})
    # Scheduler needs PassRole on LabRole; only use it once a write probe proved that works.
    # EventBridge rules invoke Lambda through a resource policy, so they need no role at all.
    scheduler = "scheduler" if st.get("scheduler.create") == OK else "rule"
    if st.get("bedrock.invoke") == OK:
        llm = {"provider": "bedrock", "model_id": ctx["bedrock_model_id"]}
    else:
        llm = {"provider": "external", "model_id": None,
               "note": "Bedrock unusable: set LLM_PROVIDER=anthropic|openai and store the key in Secrets Manager"}
    if st.get("sagemaker.endpoints") == OK and quotas.get("serverless_concurrency", 1) > 0:
        inference = "serverless-endpoint"
    elif st.get("sagemaker.training") == OK:
        inference = "batch-transform"
    else:
        inference = "model-in-lambda"
    if st.get("sagemaker.training") == OK and quotas.get("train_m5_large", 1) > 0:
        training = "sagemaker-ml.m5.large"
    else:
        training = "local"  # train on the laptop, upload the model artifact to S3
    return {
        "scheduler": scheduler,
        "llm": llm,
        "inference": inference,
        "training": training,
        "frontend_cdn": "cloudfront" if st.get("cloudfront.list") == OK else "s3-website",
        "lab_role_arn": ctx.get("lab_role_arn"),
        "learner_lab": ctx.get("learner_lab", False),
    }


# ----------------------------------------------------------------------------- output

ICON = {OK: "OK  ", DENIED: "DENY", ERROR: "ERR ", SKIPPED: "skip"}


def print_report(report: Report) -> None:
    ident = report.identity
    print(f"\nAccount {ident.get('account')}  region {report.region}  as {ident.get('arn')}")
    if report.decisions.get("learner_lab"):
        print("Detected AWS Academy Learner Lab (voclabs role): reusing LabRole, no IAM creation.")
    print()
    width = max(len(r.key) for r in report.results)
    for r in report.results:
        crit = "*" if r.critical else " "
        print(f" [{ICON[r.status]}]{crit} {r.key:<{width}}  {r.block:<38} {r.detail}")
    print("\n * = needed by the core pipeline\n")
    print("Decisions (saved to .access_report.json, read by `tasks.py deploy`):")
    for k, v in report.decisions.items():
        print(f"  {k:<13} {v}")


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    settings = Settings.from_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", default=settings.region)
    ap.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    ap.add_argument("--probe-write", action="store_true", help="create+delete tiny resources to test PassRole")
    ap.add_argument("--output", default=str(ROOT / ".access_report.json"))
    args = ap.parse_args(argv)

    session = boto3.Session(profile_name=args.profile or None, region_name=args.region)
    report = Report(generated_at=datetime.now(UTC).isoformat(timespec="seconds"), region=args.region)
    try:
        ident = client(session, "sts").get_caller_identity()
    except Exception as exc:  # noqa: BLE001
        status, detail = classify(exc)
        print(f"Cannot authenticate: {detail}")
        if isinstance(exc, ClientError) and exc.response["Error"]["Code"] in CREDENTIAL_CODES:
            print("Learner Lab credentials expire every ~4 h: copy fresh ones from 'AWS Details'.")
        print("Set credentials in ~/.aws/credentials (or AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY /"
              " AWS_SESSION_TOKEN) and rerun.")
        return 2
    report.identity = {"account": ident["Account"], "arn": ident["Arn"]}
    ctx: dict = {
        "settings": settings,
        "account": ident["Account"],
        "run_id": uuid.uuid4().hex[:8],
        "learner_lab": "voclabs" in ident["Arn"],
    }
    report.results = run_checks(session, CHECKS, ctx, args.probe_write)
    report.decisions = decide(report.results, ctx)
    print_report(report)
    Path(args.output).write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    print(f"\nReport written to {args.output}")
    failed = [r.key for r in report.results if r.critical and r.status in (DENIED, ERROR)]
    if failed:
        print("Critical checks failed: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
