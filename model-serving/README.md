# Model Serving

Using flask.

Install required packages

```bash
    pip3 install flask  flask_bootstrap  gunicorn

    # we also need tensorflow
    pip3 install tensorflow
```

To run the flask app in dev mode

```bash
    # this has a problem of printing each log entry twice! - ugh 
    $   FLASK_ENV=development FLASK_APP=app.py flask run

    # another dev mode
    $  FLASK_ENV=development python app.py
```


In production mode

```bash
    $  gunicorn --bind 0.0.0.0:5000 wsgi:app

    # using more cpu cores (workers = 2 x cores + 1)
    $  gunicorn --workers=5 --bind 0.0.0.0:8000 wsgi:app
```

Go to http://localhost:8000

For debugging

```bash
    $  docker run -u $(id -u):$(id -g)  -it --rm -v "models:/app/models" --entrypoint /bin/bash  sujee/sotu-text-generator
```

## Deploying to GCP

hostname : sotu.elephantscale.com

    ssh -i   sotu-1.pem  ubuntu@sotu.elephantscale.com


## Machine Setup

from https://docs.docker.com/engine/install/ubuntu/

```bash

$  sudo apt-get remove docker docker-engine docker.io containerd runc

$   sudo apt-get install -y \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg-agent \
        software-properties-common

$  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

$  sudo apt-key fingerprint 0EBFCD88

$  sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

$ sudo apt-get update

$ sudo apt-get install -y docker-ce docker-ce-cli containerd.io

$ sudo usermod -aG docker $(whoami)
# logout and log back in

$ docker images

$  docker run hello-world

```

### Setup SSH deploykey

deploy key is : sotu-deploy.pem

scp this key into : ~/.ssh/sotu-deploy.pem

And create a file like this:

```
# file : ~/.ssh/config
Host github.com
    HostName github.com
    PreferredAuthentications publickey
    IdentityFile ~/.ssh/sotu-deploy.pem
```

```bash
    $   ssh-keyscan github.com >> ${HOME_DIR}/.ssh/known_hosts
```

Test it like this:

```bash
    $ chmod 600 ~/.ssh/*

    $  ssh -T git@github.com
```