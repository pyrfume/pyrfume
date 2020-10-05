sudo apt update
if [ ! -d "./mysql-docker" ]; then
  mkdir mysql-docker
fi

cd mysql-docker

if [ ! -f "./docker-compose.yml" ]; then
  wget https://raw.githubusercontent.com/datajoint/mysql-docker/master/slim/docker-compose.yml
fi

if ! [ -x "$(command -v docker-compose)" ]; then
  echo 'Installing docker and docker compose...\n' >&2
  sudo apt-get remove docker docker-engine docker.io containerd runc
  sudo apt-get update
  sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common -y
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
  sudo apt-key fingerprint 0EBFCD88
  sudo add-apt-repository \
    "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs) \
    stable"
  sudo apt-get update
  sudo apt-get install docker-ce docker-ce-cli containerd.io -y
  sudo apt install libffi-dev libc-dev make python3-pip python3-dev -y
  sudo curl -L "https://github.com/docker/compose/releases/download/1.27.4/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
  sudo chmod +x /usr/local/bin/docker-compose
fi

sudo docker-compose up -d