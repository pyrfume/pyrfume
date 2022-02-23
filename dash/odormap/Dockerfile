FROM tiangolo/uwsgi-nginx-flask:python3.7
LABEL maintainer="maintainer"

##### Miniconda installation #####
RUN apt-get -qq update && apt-get -qq -y install curl bzip2 \
    && curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local \
    && rm -rf /tmp/miniconda.sh \
    && conda install -y python=3 \
    && conda update conda \
    && apt-get -qq -y remove curl bzip2 \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log \
    && conda clean --all --yes

ENV PATH /opt/conda/bin:$PATH

##### Conda packages #####
RUN pip install -U pip
RUN pip install pyrfume

#### Dash stuff ####
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY app /app

ENV NGINX_WORKER_PROCESSES auto