![](assets/header_image.png)

[![Build Main](https://github.com/ika-rwth-aachen/acdc-notebooks/actions/workflows/build.yml/badge.svg)](https://github.com/ika-rwth-aachen/acdc-notebooks/actions/workflows/build.yml)


## Start of the Assignments: [index.ipynb](index.ipynb)

__Note that the Jupyter Notebooks are sometimes not correctly displayed on Github (Missing Images).__

__Use Juypter Lab to display the notebooks.__


## Local Usage for editing/developing the notebooks with Docker
The docker solution can be used for developing and testing the notebooks. You can use the following instructions to run the system:

You can either build the docker image 
```bash
./docker_build.sh
```

or pull it from our registry:
```bash
docker pull tillbeemelmanns/acdc-notebooks:latest
```
You might need to login to Dockerhub

```bash
docker login
```

Then run `./docker_run.sh` and open the link in the terminal to open Jupyter Lab in your browser
```bash
./docker_run.sh
```
