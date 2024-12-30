FROM python:3.10-slim

COPY . /bot

WORKDIR /bot

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python", "main.py" ]