FROM python:3
COPY . /app
WORKDIR /app
RUN pip3 install -r requirements.txt

# # This makes opencv work
# RUN apt-get update
# RUN apt-get install ffmpeg libsm6 libxext6  -y

EXPOSE 8000
ENTRYPOINT ["python"]
CMD ["app.py"]
