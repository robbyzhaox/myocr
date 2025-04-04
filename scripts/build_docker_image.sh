#!/bin/bash

set +e

tag="myocr"
if docker images -q --filter "reference=*:$tag" > /dev/null; then
    echo "Removing docker image tagged: $tag ..."
    docker rmi -f $(docker images -q --filter "reference=*:$tag")
    echo "Docker image removed"
else
    echo "Do not have docker image with tag: $tag"
fi

VERSION=$(python -c 'import myocr.version; print(myocr.version.VERSION)')
echo "$VERSION"

cd ..
cp -r ~/.MyOCR/models/ ./models
export DOCKER_BUILDKIT=1
docker build \
  --progress=plain \
  --build-arg PIP_CACHE_DIR=/root/.cache/pip \
  --build-arg APT_CACHE_DIR=/var/cache/apt \
  --platform linux/amd64 -f Dockerfile-infer \
  -t myocr .
rm -rf ./models
cd -
