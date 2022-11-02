# syntax=docker/dockerfile:1
FROM python:3.8-slim-buster

WORKDIR /app

COPY . .
RUN apt-get update

RUN pip3 install -r requirements.txt \
    && python setup.py install \
    && mkdir /usr/local/share/casacore/data