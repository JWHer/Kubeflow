#Dockerfile

#FROM frolvlad/alpine-python-machinelearning
FROM ubuntu:16.04

# update pip
RUN apt-get update
RUN apt-get install -y software-properties-common vim
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update

# python, git
RUN apt-get install -y build-essential python python-dev python3.6 python3.6-dev python3-pip python3.6-venv
RUN apt-get install -y git
RUN python3.6 -m pip install pip --upgrade
RUN python3.6 -m pip install wheel

RUN python3.6 -m pip install --upgrade google-cloud-storage
RUN python3.6 -m pip install --upgrade gcsfs
RUN python3.6 -m pip install --upgrade sklearn
RUN python3.6 -m pip install --upgrade argparse
RUN python3.6 -m pip install --upgrade pandas
RUN python3.6 -m pip install --upgrade simplejson
RUN python3.6 -m pip install --upgrade transformers
#RUN python3.6 -m pip install --upgrade simpletransformers
RUN python3.6 -m pip install --upgrade torch
RUN python3.6 -m pip install --upgrade torchtext

ENV PYTHONUNBUFFERED 1
RUN mkdir -p /nlp/src
COPY . /nlp/src
WORKDIR /nlp/src

ENTRYPOINT ["python3.6", "train.py"]