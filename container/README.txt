#Build docker image from Dockerfile
docker build -t dic-assignment .

winpty docker run --rm -it -p 5000:5000 dic-assignment

curl http://localhost:5000/api/detect -F images=@./images/dogos.jpg > asdf.jpg


# Detect all images in folder (run within container)
source run_inference.sh images output