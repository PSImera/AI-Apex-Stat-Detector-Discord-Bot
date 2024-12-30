FROM python:3.9-slim

COPY . /bot

WORKDIR /bot

RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python", "main.py" ]