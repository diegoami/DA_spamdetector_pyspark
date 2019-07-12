
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline

from pyspark.sql.types import DoubleType, StringType, IntegerType
from pyspark.sql.types import StructType, StructField
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import HashingTF


from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.evaluation import MulticlassMetrics




spark = SparkSession \
    .builder.config("spark.master", "local") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

schema = StructType().add(StructField("message", StringType())).add(StructField("label", IntegerType()))
df = spark.read.option("mode", "DROPMALFORMED").schema(schema).csv("spam_out.csv")

trainingDf, testDf = df.randomSplit([0.8, 0.2], 42)
tokenizer = Tokenizer(inputCol="message", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features", numFeatures=3000)
lr = LogisticRegression(maxIter=100, regParam=0.001)

pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])
model = pipeline.fit(trainingDf)
model.write().overwrite().save("data/sparkmodel")
print("Model written to data/sparkmodel")

testPredictions = model.transform(testDf).withColumn("label", testDf["label"].cast(DoubleType()))
rdd_map = testPredictions.select("prediction", "label").rdd.map(lambda lp: (lp["prediction"], lp["label"]))
test_binary_metrics = BinaryClassificationMetrics(rdd_map)
print(f"Test PR and ROC : {test_binary_metrics.areaUnderPR}, {test_binary_metrics.areaUnderROC}\n")

test_metrics =  MulticlassMetrics(rdd_map)
print("Test Confusion matrix : ")
print(test_metrics.confusionMatrix())
print(f"Test: Precision {test_metrics.precision(1), test_metrics.precision(0)}")
print(f"Test: Recall {test_metrics.recall(1), test_metrics.recall(0)}")
print(f"Test: Accuracy {test_metrics.accuracy}")
