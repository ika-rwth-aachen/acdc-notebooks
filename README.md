![](assets/header_image.png)

[![Build Main](https://github.com/ika-rwth-aachen/acdc-notebooks/actions/workflows/build.yml/badge.svg)](https://github.com/ika-rwth-aachen/acdc-notebooks/actions/workflows/build.yml)


## Start of the Assignments: [index.ipynb](index.ipynb)

__Note that the Jupyter Notebooks are sometimes not correctly displayed on Github (Missing Images).__

__Use JuypterLab to display the notebooks.__


## Instructions to start JupyterLab with the Jupyter Notebooks using Docker
Docker should be used for starting JupyterLab on you local machine. You can use the following instructions to run the system:

You can either build the docker image 
```bash
./docker_build.sh
```

or pull it from our registry (recommended):
```bash
docker pull rwthika/acdc-notebooks:latest
```

Then navigate to `acdc-notebooks/docker` and run `./run.sh`.
```bash
./run.sh
```

The docker image should start now and your terminal should display some output. Open the last link displayed in the terminal and JupyterLab will open in your browser.

![](assets/terminal.png)

Navigate to the `/acdc/` directory in JupyterLab and open `index.ipynb` to get an overview of all available notebooks.
