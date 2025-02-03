# Dockerfile
FROM public.ecr.aws/docker/library/python:3.10-slim
 
# COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.7.1 /lambda-adapter /opt/extensions/lambda-adapter

WORKDIR /app

COPY requirements.txt  ./requirements.txt

RUN pip install -r requirements.txt

COPY app.py  ./

# Expose the port
EXPOSE 7860

CMD ["python3", "app.py"]