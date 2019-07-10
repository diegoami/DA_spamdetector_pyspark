FROM python:3
RUN apt update
RUN apt-get -y install default-jdk
RUN pip install numpy pyspark

WORKDIR /opt
RUN mkdir data

ADD create_model.py .
ADD spam_out.csv .
ADD test_model.py .
