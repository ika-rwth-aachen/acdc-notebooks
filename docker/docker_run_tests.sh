#!/bin/sh

DIR="$(cd -P "$(dirname "$0")" && pwd)"
MOUNT_DIR="$(dirname "$DIR")"

# The user inside the docker environment is `joyvan` with uid 1000
# This setting is predefined in registry.git.rwth-aachen.de/jupyter/profiles/rwth-courses:latest
# to ensure compatibility with RWTH Jupyter Hub
# We will give this user id temporarly write permissions to this directory
setfacl -R -m u:1000:rwx $MOUNT_DIR

docker run \
--name='ika-acdc-notebooks-tests' \
--gpus all \
--env CUDA_VISIBLE_DEVICES=0 \
--rm \
--interactive \
--tty \
--publish 8888:8888 \
--volume $MOUNT_DIR:/home/jovyan/acdc \
rwthika/acdc-notebooks:latest \
/bin/bash -c \
"cd /home/jovyan/acdc/section_1_introduction_and_ros &&
 time papermill 2_introduction_to_numpy_solution.ipynb /dev/null &&
 cd /home/jovyan/acdc/section_2_sensor_data_processing && 
 time papermill 1_semantic_image_segmentation_solution.ipynb /dev/null &&
 time papermill 2_augmentation_semantic_image_segmentation_solution.ipynb /dev/null &&
 time papermill 3_semantic_pcl_segmentation_solution.ipynb /dev/null &&
 time papermill 4_semantic_pcl_segmentation_boosting_solution.ipynb /dev/null &&
 time papermill 5_object_detection_solution.ipynb /dev/null &&
 time papermill 6_grid_mapping_solution.ipynb /dev/null &&
 time papermill 7_cam_semantic_grid_mapping_solution.ipynb /dev/null &&
 cd /home/jovyan/acdc/section_4_vehicle_guidance && 
 time papermill 1_route_planning_solution.ipynb /dev/null"


# Remove write permission of user 1000
setfacl -R -x u:1000 $MOUNT_DIR
