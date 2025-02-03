# Dockerfile
FROM public.ecr.aws/docker/library/python:3.10
 
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.7.0 /lambda-adapter /opt/extensions/lambda-adapter

WORKDIR /app

COPY requirements.txt  ./requirements.txt

RUN pip install -r requirements.txt

COPY model_frontend.py  ./

CMD ["python3", "model_frontend.py"]

