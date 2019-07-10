FROM python:3
RUN apt-get install default-jdk
RUN pip install numpy pyspark
RUN mkdir /data

ADD create_model.py .
ADD spam_out.csv .
ADD test_model.py .
