FROM python:3.12

WORKDIR /app
COPY . .

RUN apt-get update
RUN apt-get install -y gdal-bin libgdal-dev g++ libgl1 jpeg-dev zlib-dev libjpeg make

RUN pip install -r requirements.txt

# matplotlib config (used by benchmark)
RUN mkdir -p /root/.config/matplotlib
RUN echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc

EXPOSE 8888