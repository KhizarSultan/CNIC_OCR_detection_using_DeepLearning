# CNIC Optical Character Recognition

## Problem Statement

Detect Name, Father Name, Gender, Date of birth, Date of Issue and Date of expiry
from CNIC.

## Solution
We discover the solution of this problem using [easyocr](https://www.jaided.ai/easyocr/documentation) python library. easyocr takes an image as input and calculate the bounding boxes of text.

## Installation

```python

pip install -r requirements.txt

```
This requirements.txt file includes all necessary libraries to run this project.
## Code Exmaple

```python
import easyocr
reader = easyocr.Reader(['en'])
image_path = "data/cnic_5.jpg"
result = reader.readtext(image_path)
```
![Sample_CNIC](https://user-images.githubusercontent.com/30461028/98521241-00926e00-2295-11eb-8a95-b7f5b2bf2ae2.PNG)

## Results

![image](https://user-images.githubusercontent.com/30461028/98523050-42bcaf00-2297-11eb-9c5f-f3e84d9dc4d6.png)

you can get insights from graphs that your image size and image DPI(Dots Per Inch) is good then accuracy will be 100%.

## Deployment
We are using latest technology called Docker to deploy this app. Following are the steps and instructions before deployment:

(1) Download and Install Docker for windows from [Docker](https://docs.docker.com/docker-for-windows/) \
(2) Create a Dockerfile in Visual Studio Code using following code 
```python

# use python as base image
FROM python:3.7

RUN pip install virtualenv 
ENV VIRTUALENV=/env
RUN virtualenv venv -p python3
ENV PATH="VIRTUAL_ENV/bin:$PATH"

# RUN apt-get update ##[edited]
# RUN apt-get update
WORKDIR /app
COPY . /app
# RUN pip install skia-python==86.0
# RUN apt-get install libgl1-mesa-glx-lts-utopic libgl1-mesa-dri-lts-utopic
RUN apt-get update -y 
RUN apt install -y libsm6 libxext6
RUN apt update
RUN pip install pyglview
RUN apt install -y libgl1-mesa-glx
RUN pip install -r requirement.txt

# Expose Port
# EXPOSE 5000

#Run the application
# ENTRYPOINT python app.py
CMD ["python","app.py"]
# CMD ["flask", "run", "-h", "0.0.0.0", "-p", "5000"]
```
(3) Create the Docker Image from Dockerfile using following commands

```bash
## docker build image syntax
docker build -t <image-name>:<version> .

## for example
docker build -t cnic_extraction:latest .

``` 
(4) Save the docker image for the future use using following command

```bash

## docker save image syntax

docker save -o <path for generated tar file> <image name>

## for example
docker save -o \docker_images\cnic_extraction.tar cnic_extraction:latest

``` 
(5) Load the docker image using following command

```bash
## docker load image syntax

docker load -i <path for generated tar file>

## for example
docker load -i \docker_images\cnic_extraction.tar

``` 

(6) Run Container from docker image using following command


```bash

## docker container run syntax
docker run -p <app_port>:<docker_map_port> <image_name>

## for example
docker run -p 8080:8080 cnic_extraction

```
(7) it will run flask app at localhost:8080 in your default browser like this:

#### Input 
![image](https://user-images.githubusercontent.com/30461028/98629432-a946d900-233a-11eb-99e4-8c9173ce6177.png)

#### Output
![image](https://user-images.githubusercontent.com/30461028/98629507-d5faf080-233a-11eb-91aa-07e2acefbeb4.png)



 
## Enjoy
if you have any query regarding this project, please feel free to ask us at khizarsultan@addo.ai or usmanfarooq@addo.ai
