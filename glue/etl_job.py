"""Block 7: Glue ETL job (PySpark, Glue 4.0): raw/ JSON from Firehose -> curated/ + features/ Parquet.

  1. read new raw records (job bookmarks: each run only sees files added since the last run)
  2. clean: cast types, drop duplicates (sensor_id, ts), METR-LA zeros/nulls = missing;
     short gaps (<= 3 slots = 15 min) filled with the last valid reading (speed_filled)
  3. enrich with sensor metadata (free-flow, segment length) and recompute congestion
     index / level with the job's thresholds (same formula as the Lambda)
  4. write curated/dt=YYYY-MM-DD/ (replay date) and features/dt=.../ (hourly aggregates)
     both appended as Parquet; Athena reads them via partition projection

Arguments: --JOB_NAME --LAKE_BUCKET [--LEVEL_MODERATE 0.2 --LEVEL_HEAVY 0.4 --LEVEL_SEVERE 0.6]
"""
import sys

from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql import Window
from pyspark.sql import functions as F

REQUIRED = ["JOB_NAME", "LAKE_BUCKET"]
OPTIONAL = {"LEVEL_MODERATE": "0.2", "LEVEL_HEAVY": "0.4", "LEVEL_SEVERE": "0.6"}
present = [k for k in OPTIONAL if f"--{k}" in sys.argv]
args = getResolvedOptions(sys.argv, REQUIRED + present)
for k, v in OPTIONAL.items():
    args.setdefault(k, v)
bucket = args["LAKE_BUCKET"]
moderate, heavy, severe = (float(args[k]) for k in ("LEVEL_MODERATE", "LEVEL_HEAVY", "LEVEL_SEVERE"))

sc = SparkContext()
glue = GlueContext(sc)
spark = glue.spark_session
job = Job(glue)
job.init(args["JOB_NAME"], args)

raw_dyf = glue.create_dynamic_frame.from_options(
    connection_type="s3", format="json", transformation_ctx="raw_source",
    connection_options={"paths": [f"s3://{bucket}/raw/"], "recurse": True})
raw = raw_dyf.toDF()

if raw.rdd.isEmpty():
    print("no new raw data")
    job.commit()
    sys.exit(0)

df = (raw.select(
        F.col("sensor_id").cast("string").alias("sensor_id"),
        F.to_timestamp("ts").alias("ts"),
        F.col("speed_mph").cast("double").alias("speed_mph"),
        F.coalesce(F.col("incident_flag").cast("boolean"), F.lit(False)).alias("incident_flag"),
        F.coalesce(F.col("source").cast("string"), F.lit("unknown")).alias("source"))
      .where(F.col("sensor_id").isNotNull() & F.col("ts").isNotNull())
      .dropDuplicates(["sensor_id", "ts"])
      .withColumn("speed_mph", F.when(F.col("speed_mph") > 0, F.col("speed_mph"))))   # zeros -> missing

# gap filling: last valid value within the previous 3 slots
w = Window.partitionBy("sensor_id").orderBy(F.col("ts").cast("long")).rangeBetween(-15 * 60, 0)
df = df.withColumn("speed_filled", F.coalesce(F.col("speed_mph"), F.last("speed_mph", ignorenulls=True).over(w)))

meta = spark.read.json(f"s3://{bucket}/meta/sensor_meta.json").select(
    F.col("sensor_id").cast("string"), "free_flow_mph", "segment_length_m")
df = df.join(F.broadcast(meta), "sensor_id", "inner")

ci = F.greatest(F.lit(0.0), 1 - F.col("speed_filled") / F.col("free_flow_mph"))
df = (df.withColumn("congestion_index", F.when(F.col("speed_filled").isNotNull(), ci))
        .withColumn("level", F.when(F.col("congestion_index").isNull(), "unknown")
                    .when(F.col("congestion_index") < moderate, "free")
                    .when(F.col("congestion_index") < heavy, "moderate")
                    .when(F.col("congestion_index") < severe, "heavy").otherwise("severe"))
        .withColumn("travel_time_s", F.col("segment_length_m") / (F.col("speed_filled") * 0.44704))
        .withColumn("hour", F.hour("ts"))
        .withColumn("dow", (F.dayofweek("ts") + 5) % 7)          # Spark: Sunday=1 -> Monday=0
        .withColumn("dt", F.date_format("ts", "yyyy-MM-dd")))

curated = df.select("ts", "sensor_id", "speed_mph", "free_flow_mph", "congestion_index", "level",
                    "travel_time_s", "incident_flag", "source", "hour", "dow", "speed_filled", "dt")
curated.write.mode("append").partitionBy("dt").parquet(f"s3://{bucket}/curated/")

features = (df.groupBy("dt", "sensor_id", "dow", "hour")
              .agg(F.avg("speed_filled").alias("avg_speed_mph"),
                   F.avg("congestion_index").alias("avg_congestion"),
                   F.avg(F.when(F.col("level").isin("heavy", "severe"), 1.0).otherwise(0.0)).alias("share_congested"),
                   F.max(F.col("incident_flag").cast("int")).alias("had_incident"),
                   F.count("*").alias("readings")))
features.write.mode("append").partitionBy("dt").parquet(f"s3://{bucket}/features/")

print(f"curated rows: {curated.count()}")
job.commit()
