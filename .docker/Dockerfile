FROM python:3.12

ARG USER_ID
ARG GROUP_ID
ARG MPLCONFIGDIR=/app/.tmp/matplotlib

WORKDIR /app
COPY requirements.txt /app/

# # Create a user
RUN useradd -l -u ${USER_ID}  user
# # Chown all the files to the app user
RUN chown -R ${USER_ID}:${GROUP_ID} /app

RUN apt-get update
RUN apt-get install -y gdal-bin libgdal-dev g++ libgl1 make libx11-6 python3-tk python3-pyqt5

RUN pip install -r requirements.txt

# matplotlib config (used by benchmark)
RUN mkdir -p ${MPLCONFIGDIR}
# set env variable
ENV MPLCONFIGDIR=${MPLCONFIGDIR}

USER user

EXPOSE 8888