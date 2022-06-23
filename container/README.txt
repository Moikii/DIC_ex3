#Build docker image from Dockerfile
docker build -t dic-assignment .

# For windows run the following command to interactivly start the server on port 5000
winpty docker run --rm -it -p 5000:5000 dic-assignment

############ 1st Option ################
1st Option uploads the path (server already has the images)
This option also saves the files on the server locally without returning instead it returns the inference time. 

# Use 1st endpoint to provide path 
time curl http://localhost:5000/api/detect -d "input=images/"

# get images from Docker container (when using 1st endpoint)
docker cp <docker_name>:/app/output ./output/

########## 2nd Option ######################
2nd Option uploads the files via the api. 

# Use 2nd endpoint to upload image and receive image with bounding box
curl http://localhost:5000/api/detect/image -F images=@./images/dogos.jpg > asdf.jpg


######### 3rd Option ######################
3rd Option uploads the path (server already has the images) 
Instead of returning the inference time and saving the files locally (as Option 1 does), this returns all results as json format. 

time curl http://localhost:5000/api/detect/json -d "input=images/"


# Detect all images in folder (run within container)
time source run_inference.sh images output