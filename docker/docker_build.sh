#!/bin/sh
#docker buildx create --use
docker build \
  --platform linux/amd64 \
  --cache-from type=registry,ref=rwthika/acdc-notebooks-jupyros:latest \
  --tag rwthika/acdc-notebooks-jupyros:latest \
  .
