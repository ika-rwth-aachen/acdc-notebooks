#!/bin/sh

DIR="$(cd -P "$(dirname "$0")" && pwd)"
MOUNT_DIR="$(dirname "$DIR")"

# The user inside the docker environment is `joyvan` with uid 1000
# This setting is predefined in registry.git.rwth-aachen.de/jupyter/profiles/rwth-courses:latest
# to ensure compatibility with RWTH Jupyter Hub
# We will give this user id temporarly write permissions to this directory
setfacl -R -m u:1000:rwx $MOUNT_DIR

docker run \
--name='ika-acdc-notebooks' \
--rm \
--interactive \
--tty \
--publish 8888:8888 \
--publish 9090:9090 \
--volume $MOUNT_DIR:/home/jovyan/acdc \
rwthika/acdc-notebooks:latest

# Remove write permission of user 1000
setfacl -R -x u:1000 $MOUNT_DIR