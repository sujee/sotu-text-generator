# Pushing the Docker to Google Container Registry (GCR)

Login to GCP

Start here : https://cloud.google.com/container-registry/docs/quickstart

Enable API (select project)

Initialized Gcloud CLI

    $  gcloud auth configure-docker

Build docker image

    $  cd model-server

    $  ./build-docker.sh

Tag the docker image to GCR.IO  
Here `deep-way-265121` is PROJECT_ID

    $  docker tag  sujee/sotu-text-generator    gcr.io/deep-way-265121/sotu-text-generator:latest

Push the image
    $  docker push  gcr.io/deep-way-265121/sotu-text-generator:latest
