FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

LABEL maintainer="psimhra@gmail.com"
LABEL description="The bot is designed to read and parse information - namely statistics - from Apex Stats screenshots. It can be configured to give a user either one of two roles (or both) based on the parsed statistics"
LABEL version="1.1.0"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

RUN groupadd --gid 1000 appuser
RUN useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser
RUN apt-get update && apt-get install -y
RUN rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip install --upgrade pip setuptools wheel

WORKDIR /bot
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . .

RUN chown -R appuser:appuser /bot
USER appuser

ENTRYPOINT ["python"]
CMD ["main.py"]