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
RUN conda install -c conda-forge -y rdkit
RUN pip install -U pip

##### Create pyrfume directory #####
RUN mkdir /pyrfume
WORKDIR /pyrfume

#### Install pyrfume (root of the build context) #####
COPY setup.py ./
COPY requirements.txt ./
COPY pyrfume ./pyrfume
RUN pip install -e .

#### Dash stuff ####
COPY dash/mainland/app /app
COPY dash/mainland/requirements.txt /app/dash-requirements.txt
RUN pip install -r /app/dash-requirements.txt

ENV NGINX_WORKER_PROCESSES auto