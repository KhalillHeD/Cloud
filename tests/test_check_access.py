import check_access as ca
from botocore.exceptions import ClientError, NoCredentialsError
from traffic_common.config import Settings


def client_error(code, msg="nope"):
    return ClientError({"Error": {"Code": code, "Message": msg}}, "Op")


def test_classify():
    assert ca.classify(client_error("AccessDeniedException"))[0] == ca.DENIED
    assert ca.classify(client_error("ValidationException", "User is not authorized to perform x"))[0] == ca.DENIED
    assert ca.classify(client_error("ResourceNotFoundException"))[0] == ca.ERROR
    assert ca.classify(NoCredentialsError())[0] == ca.ERROR
    assert ca.classify(ca.Skip("later")) == (ca.SKIPPED, "later")
    html = ClientError({"Error": {"Code": "500", "Message": "<!doctype html>\n<title>500</title>"},
                        "ResponseMetadata": {"HTTPStatusCode": 500}}, "Op")
    assert ca.classify(html) == (ca.ERROR, "500: HTTP 500 (non-JSON body)")


def test_run_checks_never_crashes_and_skips_writes():
    def good(s, ctx):
        return "fine"

    def denied(s, ctx):
        raise client_error("AccessDenied")

    def boom(s, ctx):
        raise RuntimeError("kaput")

    checks = [ca.Check("a", "b", True, False, good), ca.Check("d", "b", True, False, denied),
              ca.Check("e", "b", False, False, boom), ca.Check("w", "b", True, True, good)]
    res = {r.key: r for r in ca.run_checks(None, checks, {}, probe_write=False)}
    assert res["a"].status == ca.OK and res["a"].detail == "fine"
    assert res["d"].status == ca.DENIED
    assert res["e"].status == ca.ERROR and "kaput" in res["e"].detail
    assert res["w"].status == ca.SKIPPED


def test_model_ranking_prefers_claude_haiku_profiles():
    ids = ["meta.llama3", "amazon.nova-micro-v1:0", "anthropic.claude-3-haiku-20240307-v1:0",
           "us.anthropic.claude-3-haiku-20240307-v1:0", "anthropic.claude-3-sonnet"]
    ranked = ca._rank_models(ids)
    assert ranked[0] == "us.anthropic.claude-3-haiku-20240307-v1:0"
    assert "meta.llama3" not in ranked
    assert ranked[-1] == "amazon.nova-micro-v1:0"


def _results(**statuses):
    return [ca.Result(k.replace("_", "."), "x", True, v, "") for k, v in statuses.items()]


def test_decisions_fallbacks_when_services_denied():
    r = _results(scheduler_create=ca.SKIPPED, bedrock_invoke=ca.DENIED,
                 sagemaker_endpoints=ca.DENIED, sagemaker_training=ca.DENIED, cloudfront_list=ca.DENIED)
    d = ca.decide(r, {"settings": Settings()})
    assert d["scheduler"] == "rule"
    assert d["llm"]["provider"] == "external"
    assert d["inference"] == "model-in-lambda"
    assert d["training"] == "local"
    assert d["frontend_cdn"] == "s3-website"


def test_decisions_when_everything_works():
    r = _results(scheduler_create=ca.OK, bedrock_invoke=ca.OK, sagemaker_endpoints=ca.OK,
                 sagemaker_training=ca.OK, cloudfront_list=ca.OK)
    ctx = {"bedrock_model_id": "us.anthropic.x", "sagemaker_quotas": {"serverless_concurrency": 10.0,
                                                                     "train_m5_large": 1.0}}
    d = ca.decide(r, ctx)
    assert d == {**d, "scheduler": "scheduler", "inference": "serverless-endpoint",
                 "training": "sagemaker-ml.m5.large", "frontend_cdn": "cloudfront"}
    assert d["llm"] == {"provider": "bedrock", "model_id": "us.anthropic.x"}


def test_serverless_quota_zero_means_batch_transform():
    r = _results(sagemaker_endpoints=ca.OK, sagemaker_training=ca.OK)
    d = ca.decide(r, {"sagemaker_quotas": {"serverless_concurrency": 0.0}})
    assert d["inference"] == "batch-transform"
