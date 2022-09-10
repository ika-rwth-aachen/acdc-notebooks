#!/bin/sh

DIR="$(cd -P "$(dirname "$0")" && pwd)"

# The user inside the docker environment is `joyvan` with uid 1000
# This setting is predefined in registry.git.rwth-aachen.de/jupyter/profiles/rwth-courses:latest
# to ensure compatibility with RWTH Jupyter Hub
# We will give this user id temporarly write permissions to this directory
setfacl -R -m u:1000:rwx .

docker run \
--name='ika-acdc-notebooks' \
--rm \
--interactive \
--tty \
--publish 8888:8888 \
--volume "$DIR":/home/jovyan/acdc \
registry.git.rwth-aachen.de/ika/acdc-notebooks:latest

# Remove write permission of user 1000
setfacl -R -x u:1000 .