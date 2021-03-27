#!/usr/bin/env bash

downloadFileFromGDrive() {
printf "Downloading $1\n"
curl -L -c mycookie -o temp "https://drive.google.com/uc?export=download&id=${1}"
filesize=$(wc -c temp | awk '{print $1}')
if [ $filesize -gt 10000 ]; then
  printf "Finish downloading\n"
  mv temp "${2}"
else
  content=$(cat temp)
  for (( j=0; j<$filesize-10; j++)); do
    if [ "${content:$j:8}" == "confirm=" ]; then
      for (( k=0; k<10; k++)); do
        if [ "${content:$j+8+$k:1}" == "&" ]; then
          token=${content:$j+8:$k}
        fi
      done
    fi
  done
  printf "Confirm downloading with token ${token}\n"
  curl -L -b mycookie -o "${2}" "https://drive.google.com/uc?export=download&confirm=${token}&id=${1}"
  rm mycookie
  rm temp
fi
}

CHECKPOINT=checkpoints/abc_10K_dataset
mkdir -p $CHECKPOINT
#gets the pretrained weights
downloadFileFromGDrive 1wXPwDGutSW2d-oa9UORPkoDITEOU_wIm abc_10K_weights.tar.gz
tar -xzvf abc_10K_weights.tar.gz && rm abc_10K_weights.tar.gz
mv latest_net.pth $CHECKPOINT
echo "downloaded pretrained weights to" $CHECKPOINT