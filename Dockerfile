FROM python:3
COPY . /app
WORKDIR /app
RUN pip3 install -r requirements-prod.txt

EXPOSE 8000
ENTRYPOINT ["python"]
CMD ["app.py"]
