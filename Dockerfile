FROM python:3.7-slim
WORKDIR /usr/src/app


#Check for SUDO, since fastai_audio install script uses SUDO
RUN if type sudo 2>/dev/null; then \
     echo "The sudo command already exists... Skipping."; \
    else \
     echo -e "#!/bin/sh\n\${@}" > /usr/sbin/sudo; \
     chmod +x /usr/sbin/sudo; \
    fi

RUN apt update && apt install -y \
  git \
  sox \
  libsox-dev \
  libsox-fmt-all \
  python3-dev \
  gcc \
  build-essential

ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install https://download.pytorch.org/whl/cpu/torch-1.1.0-cp37-cp37m-linux_x86_64.whl
RUN pip install https://download.pytorch.org/whl/cpu/torchvision-0.3.0-cp37-cp37m-linux_x86_64.whl
RUN pip install fastai

#fastai_audio install
RUN git clone https://github.com/mogwai/fastai_audio && \
  fastai_audio/install.sh && \
  mv fastai_audio/audio . && \
  rm -rf fastai_audio/

ADD app.py /usr/src/app
COPY /models /usr/src/app/models
COPY 7c54cd2c-5984-44b3-af1d-3f8a9bc5129e.key /usr/src/app/7c54cd2c-5984-44b3-af1d-3f8a9bc5129e.key
COPY amd-recordings-ccb5aa3253dd.json /usr/src/app/amd-recordings-ccb5aa3253dd.json
#COPY /sample_files /usr/src/app/sample_files
COPY /test_files /usr/src/app/test_files
COPY .env /usr/src/app/.env
EXPOSE 8000
CMD [ "python", "./app.py", "debug"]
