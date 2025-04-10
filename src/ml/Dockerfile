FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04


ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y --no-install-recommends \
    software-properties-common curl

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt install -y --no-install-recommends \
    python3.10 \
    python3.10-distutils

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

RUN curl -Ss https://bootstrap.pypa.io/get-pip.py | python3

RUN python3 -m pip install --upgrade pip  \
    && python3 -m pip install setuptools --no-cache-dir

COPY requirements.txt .

RUN python3 -m pip install -r requirements.txt --no-cache-dir