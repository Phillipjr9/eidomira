FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y python3.11 python3-pip libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements-gpu.txt ./
RUN python3.11 -m pip install --no-cache-dir -r requirements-gpu.txt
COPY . .
EXPOSE 8000
CMD ["python3.11","-m","uvicorn","app.main:app","--host","0.0.0.0","--port","8000","--proxy-headers"]
