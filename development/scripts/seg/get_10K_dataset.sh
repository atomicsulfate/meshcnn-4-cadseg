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

DATADIR='datasets' #location where data gets downloaded to

echo "downloading the data and putting it in: " $DATADIR
mkdir -p $DATADIR && cd $DATADIR
downloadFileFromGDrive 1h37RbFa0D4SLGfiDNwxlfB1BbVG_L4k_ abc_10K_dataset.tar.gz
tar -xzvf abc_10K_dataset.tar.gz && rm abc_10K_dataset.tar.gz


