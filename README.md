![](assets/header_image.png)

[![Build Main](https://github.com/ika-rwth-aachen/acdc-notebooks/actions/workflows/build.yml/badge.svg)](https://github.com/ika-rwth-aachen/acdc-notebooks/actions/workflows/build.yml)


## Start of the Assignments -> [index.ipynb](index.ipynb)

__Note that the Jupyter Notebooks are sometimes not correctly displayed on Gitlab (Missing Images). Use Juypter Lab to display the notebooks.__


## Local Usage for editing/developing the notebooks with Docker
The docker solution can be used for developing and testing the notebooks. You can use the following instructions to run the system:

You can either build the docker image 
```bash
./docker_build.sh
```

or pull it from the registry
```bash
docker pull registry.git.rwth-aachen.de/ika/acdc-notebooks:latest
```
. In both cases you will need to login to the RWTH gitlab

```bash
docker login registry.git.rwth-aachen.de
```

Then run it and open the link to open the environment in your browser
```bash
./docker_run.sh
```


## Local Usage for editing/developing the notebooks with a virtual environment
For the local development of the notebooks with a virtual environment you will have to create an virtual environment for the Python packages:

```bash
python3 -m venv my_venv
```

```bash
source my_venv/bin/activate
```

You will have to install all the requirements
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

and __also__ 
```bash
pip install jupyterlab
```

You can then simply run
```bash
jupyter lab
```
in the root directory of this repo and the environment will open automatically in your browser.