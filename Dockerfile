FROM python:3.9-slim

WORKDIR /root

COPY . .

RUN pip install -r requirements.txt

#CMD ["python", "prep_docs.py"]
