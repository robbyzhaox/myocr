#!/bin/bash

set +e

# Check for argument
if [ -z "$1" ] || { [ "$1" != "cpu" ] && [ "$1" != "gpu" ]; }; then
    echo "Usage: $0 [cpu|gpu]"
    exit 1
fi

TARGET_ENV=$1

# Determine Dockerfile and base tag based on argument
if [ "$TARGET_ENV" = "gpu" ]; then
    DOCKERFILE="Dockerfile-infer-GPU"
    BASE_TAG="myocr:gpu"
    echo "Building GPU image..."
elif [ "$TARGET_ENV" = "cpu" ]; then
    DOCKERFILE="Dockerfile-infer-CPU"
    BASE_TAG="myocr:cpu"
    echo "Building CPU image..."
fi

VERSION=$(python -c 'import myocr.version; print(myocr.version.VERSION)')
IMAGE_TAG="${BASE_TAG}-${VERSION}"
echo "Building version: $VERSION"
echo "Image tag: $IMAGE_TAG"

# Stop and remove existing containers for this specific image tag
container_ids=$(docker ps -a --filter "ancestor=$IMAGE_TAG" -q)
if [ -n "$container_ids" ]; then
    echo "Stopping and removing existing containers for $IMAGE_TAG..."
    docker stop $container_ids
    docker rm $container_ids
    echo "Containers removed"
fi

# Remove existing image with the specific tag
if docker images -q $IMAGE_TAG | grep -q .; then
    echo "Removing existing docker image: $IMAGE_TAG ..."
    docker rmi -f $IMAGE_TAG
    echo "Docker image removed"
else
    echo "No existing docker image found with tag: $IMAGE_TAG"
fi

# Copy models (ensure this path is correct)
echo "Copying models..."
cp -r ~/.MyOCR/models/ ./models

# Build the image
echo "Building image using $DOCKERFILE..."
export DOCKER_BUILDKIT=1
docker build \
  --progress=plain \
  --build-arg PIP_CACHE_DIR=/root/.cache/pip \
  --build-arg APT_CACHE_DIR=/var/cache/apt \
  --platform linux/amd64 -f $DOCKERFILE \
  -t $IMAGE_TAG .

# Clean up copied models
echo "Cleaning up models directory..."
rm -rf ./models

echo "Image build complete: $IMAGE_TAG"

# Optional: Run the container (consider if this should be separate)
echo "Running container $IMAGE_TAG..."
docker run -d -p 8000:8000 $IMAGE_TAG

echo "Script finished successfully."
