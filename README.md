SPAM DETECTOR using PY SPARK
=========================

A model created in PySpark to classify mails as spam or not spam.
This project was meant to verify the compatibility of models between Pyspark and Java-Spark.
The models are saved in the directory `data/sparkmodel`

The related project in Java Spark is here: http://github.com/diegoami/DA_spamdetector_javaspark


## SET UP

In a Python environment, install Pyspark and numpy 

## RUN locally

To create the model and export it to `data/sparkmodel`, execute the command

```
python create_model.py
```

This model is the same generated by http://github.com/diegoami/DA_spamdetector_javaspark

To test it

```
python test_model.py
```