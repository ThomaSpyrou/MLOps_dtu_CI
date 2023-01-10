# # Base image
# FROM python:3.7-slim

# # install python
# RUN apt update && \
#     apt install --no-install-recommends -y build-essential gcc && \
#     apt clean && rm -rf /var/lib/apt/lists/*

# COPY requirements.txt requirements.txt
# COPY archived/ archived/
# COPY config/ config/
# COPY src/ src/
# COPY data/ data/

# WORKDIR /
# RUN pip install -r requirements.txt --no-cache-dir
# RUN pip3 install hydra-core --upgrade
# RUN pip3 install wget

# ENTRYPOINT ["python", "-u", "src/models/train_model.py"]
FROM busybox
CMD echo "Howdy cowboy"
