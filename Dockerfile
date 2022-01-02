FROM debian:10.3-slim
COPY . /app
WORKDIR /app
RUN apt-get update && apt-get -y dist-upgrade
RUN apt-get -y install apt-utils \
    build-essential \
    python3 \
    gcc \
    python3-dev \
    python3-pip \
    python3-numpy \
    python3-pandas
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
EXPOSE 5001 5000
ENTRYPOINT [ "python3" ]
CMD ["boston_housing/app.py"]