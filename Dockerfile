FROM ubuntu:18.04

# python & build essentials
RUN apt-get update
RUN apt-get install -y --fix-missing python3-minimal python3-pip python3-dev 
RUN apt-get install -y git
RUN pip3 install -U pip

# copy repo & install dependencies
RUN mkdir -p ml_heat
COPY . /ml_heat
WORKDIR ml_heat
RUN pip3 install -Ur requirements_preprocessing.txt
RUN python3 setup_preprocessing.py develop
RUN mkdir ml_heat/__data_store__