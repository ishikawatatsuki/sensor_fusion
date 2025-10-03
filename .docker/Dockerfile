FROM python:3.12

# Install system packages for GDAL/Fiona
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Optional: make sure GDAL version is visible to pip
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

WORKDIR /app
COPY requirements.txt /app/

RUN pip install -r requirements.txt

EXPOSE 8888