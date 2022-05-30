#Build docker image from Dockerfile
docker build -t dic-assignment .

winpty docker run --rm -it -p 5000:5000 dic-assignment

curl http://localhost:5000/api/detect -F images=@./container/images/dogos.jpg > asdf.jpg