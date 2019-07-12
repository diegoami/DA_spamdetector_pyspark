
from pyspark.sql import SparkSession

from pyspark.ml import PipelineModel
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import DoubleType, StringType, IntegerType


from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.evaluation import MulticlassMetrics


spark = SparkSession \
    .builder.config("spark.master", "local") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

schema = StructType().add(StructField("message", StringType())).add(StructField("label", IntegerType()))
df = spark.read.option("mode", "DROPMALFORMED").schema(schema).csv("spam_out.csv")


loaded_model = PipelineModel.load("data/sparkmodel")
schemaPred = StructType().add("message", "string")

rowDf = spark.createDataFrame([
    Row("Winner! You have won a car"),
    Row("I feel bad today"),
    Row("Please call our customer service representative"),
    Row("Your free ringtone is waiting to be collected. Simply text the password")], schemaPred)

predictions_loaded = loaded_model.transform(rowDf)
print(predictions_loaded)
result = predictions_loaded.select(["message", "probability", "prediction"]).collect()

for row in result:
    print(row.message, row.probability, row.prediction)

predictions = loaded_model.transform(df).withColumn("label", df["label"].cast(DoubleType()))

rdd_map = predictions.select("prediction", "label").rdd.map(lambda lp: (lp["prediction"], lp["label"]))
binary_metrics = BinaryClassificationMetrics(rdd_map)
print(f"Overall PR and ROC : {binary_metrics.areaUnderPR}, {binary_metrics.areaUnderROC}\n")

multiclass_metrics = MulticlassMetrics(rdd_map)

print("Confusion matrix : ")
print(multiclass_metrics.confusionMatrix())
print(f"Overall: Precision {multiclass_metrics.precision(1), multiclass_metrics.precision(0)}")
print(f"Overall: Recall {multiclass_metrics.recall(1), multiclass_metrics.recall(0)}")
print(f"Overall Accuracy {multiclass_metrics.accuracy}")
