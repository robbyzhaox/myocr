#!/bin/bash

set +e

tag="myocr"

container_ids=$(docker ps -a --filter "ancestor=myocr:myocr" -q)

if [ -n "$container_ids" ]; then
    docker stop $container_ids
    docker rm $container_ids
    echo "Docker removed"
fi

if docker images -q --filter "reference=*:$tag" > /dev/null; then
    echo "Removing docker image tagged: $tag ..."
    docker rmi -f $(docker images -q --filter "reference=*:$tag")
    echo "Docker image removed"
else
    echo "Do not have docker image with tag: $tag"
fi

VERSION=$(python -c 'import myocr.version; print(myocr.version.VERSION)')
echo "$VERSION"

cp -r ~/.MyOCR/models/ ./models
export DOCKER_BUILDKIT=1
docker build \
  --progress=plain \
  --build-arg PIP_CACHE_DIR=/root/.cache/pip \
  --build-arg APT_CACHE_DIR=/var/cache/apt \
  --platform linux/amd64 -f Dockerfile-infer-GPU \
  -t myocr:myocr .
rm -rf ./models

docker run -d -p 8000:8000  myocr:myocr
