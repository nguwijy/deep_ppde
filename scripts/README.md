# Bash scripts for deep branching solver
## build_docker.sh
To build the container with proper permission,
make sure you run the following before running `start_docker.sh`:
```bash
bash build_docker.sh
```

## start_docker.sh
To avoid installing all python dependencies
of deep branching solver on your machine,
it is recommended to use `docker`.
All you have to do is to
[install docker](https://docs.docker.com/engine/installation/)
and run:
```bash
bash start_docker.sh
```
This will direct you to the docker environment,
where you can start the jupyter server using:
```bash
jupyter notebook
```
If you do not have GPU in your machine,
remove the line `--gpus all`
in `start_docker.sh` before running the script.
and change the first line in Dockerfile
from
`tensorflow/tensorflow:latest-gpu`
to
`tensorflow/tensorflow:latest`.
