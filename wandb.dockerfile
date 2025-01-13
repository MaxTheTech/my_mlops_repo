FROM python:3.10-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN pip install wandb

COPY src/ src/

ENTRYPOINT ["python", "-u", "src/my_mlops_project/wandb_tester.py"]
