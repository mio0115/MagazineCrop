FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /workspace
COPY requirements_dev.txt .
RUN pip install --no-cache-dir -r requirements_dev.txt

COPY . .
CMD ["python", "-m", "src.train.train"]