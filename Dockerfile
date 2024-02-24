ARG PYTORCH="1.12.1"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN apt-get update \
    && apt-get install -y git ninja-build libsparsehash-dev \
    cmake build-essential \
    glibc-source libstdc++6

SHELL ["/bin/bash", "-c"]

RUN DEBIAN_FRONTEND=noninteractive TZ=ETC/UTC apt-get install -y libpcl-dev && pip install python-pcl

COPY ./requirements.txt /workspace
#WORKDIR /workspace

# Remove non-docker dependencies
RUN pip install -r requirements.txt
RUN pip install spconv-cu113
# RUN pip install -e .

CMD ["python"]
