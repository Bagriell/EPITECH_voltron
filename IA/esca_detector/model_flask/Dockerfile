FROM tensorflow/tensorflow:latest-gpu-py3
# RUN apk add zlib-dev jpeg-dev gcc musl-dev linux-headers g++
COPY . .
# RUN pip3 show tensoflow
# RUN pip3 --version
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install --upgrade tensorflow
EXPOSE 5000
ENTRYPOINT [ "python3" ]
CMD [ "app.py" ]