#!/bin/bash
echo 'Downloading and setting up model'
DEST_DIR='bert/weight' 
FILENAME='bert-nli.zip'

if [ -d "$DEST_DIR" ]; then
echo "Directory already exists" ;
else
`mkdir -p $DEST_DIR`;

gdown https://drive.google.com/uc?id=16xHxgmzGaJaG6MlH7GjtGgCe_3BfHjuX
mv $FILENAME $DEST_DIR
unzip "${DEST_DIR}/${FILENAME}" -d $DEST_DIR
rm "${DEST_DIR}/${FILENAME}"
echo 'Done'
fi 