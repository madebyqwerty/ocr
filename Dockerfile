
FROM python:3.11.3-alpine

WORKDIR /app

RUN apk update && apk add --no-cache \
    build-base \
    musl-dev \
    linux-headers \
    g++ \
    tesseract-ocr \
    tesseract-ocr-data-ces

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

COPY . .

EXPOSE 3001

CMD ["python3", "main.py"]