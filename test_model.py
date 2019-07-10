
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.ml import PipelineModel
from pyspark.sql import Row
from pyspark.sql.types import StructType



spark = SparkSession \
    .builder.config("spark.master", "local") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

loadedModel = PipelineModel.load("data/sparkmodel")
schemaPred = StructType().add("message", "string")

rowDf = spark.createDataFrame([
    Row("Winner! You have won a car"),
    Row("I feel bad today"),
    Row("Please call our customer service representative"),
    Row("Your free ringtone is waiting to be collected. Simply text the password")], schemaPred)

predictionsLoaded = loadedModel.transform(rowDf)
print(predictionsLoaded)
result = predictionsLoaded.select(["message", "probability", "prediction"]).collect()

for row in result:
    print(row.message, row.probability, row.prediction)