FROM python:3.11-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements_dummy.txt requirements_dummy.txt
COPY src/my_mlops_project/main_dummy.py main_dummy.py

WORKDIR /
RUN pip install -r requirements_dummy.txt --no-cache-dir

ENTRYPOINT ["python", "-u", "main_dummy.py"]
