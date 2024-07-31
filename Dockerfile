FROM python:3.12

ARG USER_ID
ARG GROUP_ID

WORKDIR /app
COPY requirements.txt /app/

# # Create a user
RUN useradd -l -u ${USER_ID}  user
# # Chown all the files to the app user
RUN chown -R ${USER_ID}:${GROUP_ID} /app

RUN apt-get update
RUN apt-get install -y gdal-bin libgdal-dev g++ libgl1 make

RUN pip install -r requirements.txt

# matplotlib config (used by benchmark)
RUN mkdir -p /root/.config/matplotlib
RUN echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc

USER user

EXPOSE 8888