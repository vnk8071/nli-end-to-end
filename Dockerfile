FROM python:3.8-slim
WORKDIR /nli

RUN apt-get update

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . /nli

EXPOSE 5000
CMD ["python", "./app/views.py"]