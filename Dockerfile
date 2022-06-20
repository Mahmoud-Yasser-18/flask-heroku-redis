FROM python:3.9.13-slim

WORKDIR /app

RUN apt-get update
RUN apt update
RUN apt-get install -y unzip python3-pip wget 

#RUN apk add --no-cache gcc musl-dev linux-headers
RUN apt-get install libsndfile1 -y
COPY requirements.txt .
RUN pip install -r requirements.txt
ENV PORT 8080

COPY server/ /app
WORKDIR /app
COPY . .
CMD exec gunicorn --bind :$PORT --workers 2 app:app