![](assets/header_image.png)

[![Build Main](https://github.com/ika-rwth-aachen/acdc-notebooks/actions/workflows/build.yml/badge.svg)](https://github.com/ika-rwth-aachen/acdc-notebooks/actions/workflows/build.yml)


## Start of the Assignments: [index.ipynb](index.ipynb)

__Note that the Jupyter Notebooks are sometimes not correctly displayed on Github (Missing Images).__

__Use JuypterLab to display the notebooks.__


## Instruction to start JupyterLab with the Jupyter Notebooks using Docker
Docker should be used for starting JupyterLab on you local machine. You can use the following instructions to run the system:

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

The docker image should start now and you will get a bunch of output. Then klick on the link in the terminal and JupyterLab will open in your browser.

![](assets/terminal.png)

change to `/acdc/` directory and open `index.ipynb` to get an overview over all available notebooks.