
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline

from pyspark.sql.types import DoubleType, StringType, IntegerType
from pyspark.sql.types import StructType, StructField
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import HashingTF


from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.evaluation import MulticlassMetrics

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator



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

testPredictions = model.transform(testDf).withColumn("label", testDf["label"].cast(DoubleType()))
predictions = model.transform(df).withColumn("label", df["label"].cast(DoubleType()))

testMetrics = BinaryClassificationMetrics(testPredictions.select("prediction", "label").rdd.map(lambda lp: (lp["prediction"], lp["label"])))
print(f"Test PR and ROC : {testMetrics.areaUnderPR}, {testMetrics.areaUnderROC}\n")

overallMetrics = BinaryClassificationMetrics(predictions.select("prediction", "label").rdd.map(lambda lp: (lp["prediction"], lp["label"])))
print(f"Overall PR and ROC : {overallMetrics.areaUnderPR}, {overallMetrics.areaUnderROC}\n")

metrics = MulticlassMetrics(predictions.select("prediction", "label").rdd.map(lambda lp: (lp["prediction"], lp["label"])))
confusion = metrics.confusionMatrix()

print("Confusion matrix : ")
print(confusion)
model.write().overwrite().save("data/sparkmodel")
print("Model written to data/sparkmodel")