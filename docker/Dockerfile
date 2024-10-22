FROM jupyter/minimal-notebook:lab-3.1.18

USER root

# Install all OS dependencies for fully functional notebook server
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    vim-tiny \
    git \
    inkscape \
    libsm6 \
    libxext-dev \
    libxrender1 \
    lmodern \
    netcat \
    openssh-client \
    # ---- nbconvert dependencies ----
    texlive-xetex \
    texlive-fonts-recommended \
    texlive-plain-generic \
    # ----
    tzdata \
    unzip \
    nano-tiny && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create alternative for nano -> nano-tiny
RUN update-alternatives --install /usr/bin/nano nano /bin/nano-tiny 10

RUN apt-get update && \
	apt install -y git

USER ${NB_USER}

RUN pip install \
	'jupyterlab_iframe==0.4.0' \
	'git+https://github.com/stv0g/nbgitpuller@f735265f7b2a429a17a8fab70cfd3557f060640d' \
	'rwth-nb==0.1.8' \
	'tqdm'

RUN conda install --name base --no-update-deps \
		'conda==4.10.3' && \
	conda install --name base --quiet --yes --no-update-deps \
		'jupyterlab-git==0.32.1' \
		'ujson==4.0.2' \
		'ipympl==0.9.3' \
		'matplotlib-base==3.4.3' && \
	conda clean --all


RUN conda install --name base --quiet --yes --no-update-deps \
		'jupyterlab-drawio==0.9.0' \
		'jupyterlab_iframe==0.4.0' \
		'jupyterlab_latex=3.0.0' \
		'jupyter_core==4.11.2' && \
	conda clean --all

RUN conda install --name base --quiet --yes --no-update-deps --channel conda-forge \
                'jupyterlab_widgets' \
		'jupyterhub=1.1.0' \
		'ipywidgets' && \
        conda clean --all

# Fix traitlets pip package
RUN pip uninstall -y traitlets
RUN pip install traitlets==5.9.0

# Workaround LaTex
RUN git clone https://github.com/joequant/jupyterlab-latex /tmp/jupyterlab-latex && cd /tmp/jupyterlab-latex && git checkout cbb66825786ecf11a35fd92df797f8ccee719ad4 && pip install -ve .
RUN jupyter lab build


USER root

# Use RWTH Mirror
#RUN sed -i 's|http://archive.ubuntu.com|http://ftp.halifax.rwth-aachen.de|g' /etc/apt/sources.list

RUN apt-get update && \
	apt-get -y install \
		language-pack-de \
		texlive-latex-recommended \
		openssh-client && \
	rm -rf /var/lib/apt/lists/*

# Add more locales
RUN locale-gen de_DE && \
	locale-gen de_DE.UTF-8 && \
	update-locale

USER ${NB_USER}

ENV JUPYTER_ENABLE_LAB=yes

# Until HERE RWTH BASE IMAGE 

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


# Install ROS
RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                       ros-noetic-ros-base \
                       ros-noetic-rospack \
                       ros-noetic-ros-core  \
                       ros-noetic-rospy \
                       ros-noetic-rosbag  \
                       ros-noetic-rosauth \
                       ros-noetic-catkin \
                       ros-noetic-rosconsole \
                       ros-noetic-franka-ros \
                       ros-noetic-interactive-marker-tutorials \
                       ros-noetic-teb-local-planner \
                       ros-noetic-turtlebot3 \
                       ros-noetic-turtlebot3-fake \
                       ros-noetic-mrt-cmake-modules \
                       ros-noetic-lanelet2 \
                       ros-noetic-rosbridge-server \
                       ros-noetic-rosbridge-suite \
                       ros-noetic-tf2-web-republisher \
                       ros-noetic-actionlib \
                       libboost-dev \
                       libeigen3-dev \
                       libgeographic-dev \
                       libpugixml-dev \
                       libboost-python-dev \
                       libpython3-dev \
                       python3-catkin-tools \
                       python3-catkin-pkg \
                       python3-rospkg \
                       python3-empy \
                       python3-osrf-pycommon \
                       python3-wstool \
    && rm -rf /var/lib/apt/lists/*


# Install Lanelet2 and build ROS workspace
RUN git clone https://github.com/fzi-forschungszentrum-informatik/Lanelet2 /lanelet2 && \
    cd /lanelet2 && \
    source /opt/ros/noetic/setup.sh && \
    catkin config --source-space /lanelet2 -DPYTHON_EXECUTABLE=/usr/bin/python3 --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DPYTHON_VERSION=3.8 && \
    catkin build

USER jovyan

ARG TARGETPLATFORM

RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
    pip install tensorflow==2.10.0 ; \
    elif [ "$TARGETPLATFORM" = "linux/arm64" ] ; then \
    pip install tensorflow==2.10.0 ; \
    fi

# Install packages via requirements.txt
ADD requirements.txt .
RUN pip install -r requirements.txt

# Install Plotly Widget for PCL visualization
RUN pip install plotly

# Install PointPillars Package
RUN pip install git+https://github.com/ika-rwth-aachen/PointPillars.git

# Install Jupyros
RUN conda install jupyter bqplot pyyaml ipywidgets ipycanvas
RUN pip install jupyros==0.7.0a0
RUN pip install roslibpy
RUN pip install pyrosbag

# install the zethus and urdf libraries
RUN pip install pyOpenSSL --upgrade
RUN pip install pymongo 

RUN pip install jupyterlab-urdf==0.2.1
RUN pip install jupyterlab-zethus
RUN pip install numpy==1.25.2

ENV ROS_HOSTNAME=localhost
ENV ROS_MASTER_URI=http://localhost:11311