FROM python:3.12

WORKDIR /app
COPY . .

RUN apt-get update
RUN apt-get install -y gdal-bin libgdal-dev g++

RUN python -m venv /opt/venv
# Enable venv
ENV PATH="/opt/venv/bin:$PATH"

# RUN pip3 install -Ur requirements.txt

# RUN sh raw_data_downloader.sh