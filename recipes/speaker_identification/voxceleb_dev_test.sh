#!/bin/bash

URL=https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox2/
FILEBASE=vox2_dev_aac_parta
TESTFILE=vox2_test_aac.zip

for letter in a b c d e f g h; do
  FILE_URL=${URL}${FILEBASE}${letter}
  echo $FILE_URL
  wget $FILE_URL
done

# concatenate into 1 zip archive file
cat ${FILEBASE}* > vox2_acc.zip

# remove partial files
rm ${FILEBASE}*