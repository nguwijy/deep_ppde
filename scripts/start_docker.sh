cd $(dirname "$0")
username=$(grep -r "ARG username" Dockerfile | sed "s/ARG username=//")
cd ../
# remove the line --gpus all \ if you do not have GPU
docker run -p 8888:8888 \
    --gpus all \
    --rm -it \
    -v ${PWD}:/home/${username}/deep_ppde \
    --network host \
    nguwijy/deep_ppde \
    bash
