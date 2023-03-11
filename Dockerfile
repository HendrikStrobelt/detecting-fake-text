FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /app
COPY requirements.txt /app
RUN pip install -r requirements.txt

COPY . /app
RUN python preload_gpt2.py

CMD ["python", "server.py","--address", "0.0.0.0", "--port", "8000"]

