#!/bin/sh

docker buildx build \
--push \
--platform linux/amd64,linux/arm64 \
--cache-from type=registry,ref=rwthika/acdc-notebooks:latest \
--tag rwthika/acdc-notebooks:latest \
.
