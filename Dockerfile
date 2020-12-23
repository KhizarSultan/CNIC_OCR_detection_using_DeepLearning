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
