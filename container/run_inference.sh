#!/bin/bash 

InputFolder=$1
OutputFolder=$2

mkdir -p $OutputFolder

for FILE in "$InputFolder"/*; do 
    filename=$(basename -- "$FILE")
    filename="${filename%.*}"
    echo "$OutputFolder/$filename.jpg"
    curl http://localhost:5000/api/detect/image -F images=@"$FILE" > "$OutputFolder/$filename.jpg"
done