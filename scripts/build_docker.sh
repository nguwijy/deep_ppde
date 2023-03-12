cd $(dirname "$0")
cp ../requirements.txt ./
sed -i "/ARG username=/c ARG username=$USER" Dockerfile
sed -i "/ARG userid=/c ARG userid=$UID" Dockerfile
docker build --rm -t nguwijy/deep_ppde . -f Dockerfile
rm requirements.txt
