
FROM python:3.11.0-alpine

WORKDIR /app

RUN apk update && apk add --no-cache \
    tesseract-ocr \
    tesseract-ocr-data-ces

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

EXPOSE 3001

CMD ["python3", "main.py"]

#sudo docker build . -t ocr-service
#sudo docker run -it ocr-service /bin/sh