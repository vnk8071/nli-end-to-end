#!/bin/bash
echo 'Downloading and setting up model'
DEST_DIR='roberta/weight' 
FILENAME='weight.zip'

if [ -d "$DEST_DIR" ]; then
echo "Directory already exists" ;
else
`mkdir -p $DEST_DIR`;

gdown https://drive.google.com/uc?id=1_Xl2XkVdyoKn3EYYKt8zMjTEJFrvsl0r
mv $FILENAME $DEST_DIR
unzip "${DEST_DIR}/${FILENAME}" -d $DEST_DIR
rm "${DEST_DIR}/${FILENAME}"
echo 'Done'
fi 