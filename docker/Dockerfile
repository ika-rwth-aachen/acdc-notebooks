# Specify parent image. Please select a fixed tag here.
ARG BASE_IMAGE=rwthika/acdc-notebooks:rwth-courses
FROM ${BASE_IMAGE}

# Install cv2
USER root

# Install Essentials + cv2
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        cmake \
        build-essential \
        curl \
        wget \
        gnupg2 \
        lsb-release \
        ca-certificates \
        python3-opencv \
    && rm -rf /var/lib/apt/lists/*

RUN pip install opencv-python

# Install ROS
RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                       ros-noetic-ros-base \
                       ros-noetic-rospack \
                       ros-noetic-catkin \
                       ros-noetic-mrt-cmake-modules \
                       ros-noetic-lanelet2 \
                       libboost-dev \
                       libeigen3-dev \
                       libgeographic-dev \
                       libpugixml-dev \
                       libboost-python-dev \
                       python3-catkin-tools \
                       python3-empy \
    && rm -rf /var/lib/apt/lists/*

# Install Lanelet2 and build ROS workspace
RUN git clone https://github.com/fzi-forschungszentrum-informatik/Lanelet2 /lanelet2 && \
    cd /lanelet2 && \
    source /opt/ros/noetic/setup.sh && \
    catkin config --source-space /lanelet2 -DPYTHON_EXECUTABLE=/usr/bin/python3 && \
    catkin build

USER jovyan

# Install packages via requirements.txt
ADD requirements.txt .
RUN pip install -r requirements.txt

# Install Plotly Widget for PCL visualization
RUN jupyter labextension install jupyterlab-plotly

# Install PointPillars Package
RUN pip install git+https://github.com/ika-rwth-aachen/PointPillars.git@fix/ika-changes

# Install TensorBoard Widget
RUN pip install git+https://github.com/cliffwoolley/jupyter_tensorboard.git
RUN pip install git+https://github.com/chaoleili/jupyterlab_tensorboard.git
RUN jupyter tensorboard enable --user
