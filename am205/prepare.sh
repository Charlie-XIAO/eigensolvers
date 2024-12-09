#!/bin/bash

# Prepare the datasets

mkdir -p data

BASE_URL="https://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing"

SET="bcsstruc1"
for i in $(seq -w 1 13); do
  FILE_NAME="bcsstk${i}.mtx.gz"
  echo "Downloading ${FILE_NAME}..."
  wget -q "${BASE_URL}/${SET}/${FILE_NAME}" -P data/

  echo "Unzipping ${FILE_NAME}..."
  gunzip -f "data/${FILE_NAME}"
done

SET="bcsstruc2"
for i in $(seq -w 14 18); do
  FILE_NAME="bcsstk${i}.mtx.gz"
  echo "Downloading ${FILE_NAME}..."
  wget -q "${BASE_URL}/${SET}/${FILE_NAME}" -P data/

  echo "Unzipping ${FILE_NAME}..."
  gunzip -f "data/${FILE_NAME}"
done

SET="bcsstruc3"
for i in $(seq -w 19 25); do
  FILE_NAME="bcsstk${i}.mtx.gz"
  echo "Downloading ${FILE_NAME}..."
  wget -q "${BASE_URL}/${SET}/${FILE_NAME}" -P data/

  echo "Unzipping ${FILE_NAME}..."
  gunzip -f "data/${FILE_NAME}"
done

SET="bcsstruc4"
for i in $(seq -w 26 28); do
  FILE_NAME="bcsstk${i}.mtx.gz"
  echo "Downloading ${FILE_NAME}..."
  wget -q "${BASE_URL}/${SET}/${FILE_NAME}" -P data/

  echo "Unzipping ${FILE_NAME}..."
  gunzip -f "data/${FILE_NAME}"
done

SET="bcsstruc5"
for i in $(seq -w 29 33); do
  FILE_NAME="bcsstk${i}.mtx.gz"
  echo "Downloading ${FILE_NAME}..."
  wget -q "${BASE_URL}/${SET}/${FILE_NAME}" -P data/

  echo "Unzipping ${FILE_NAME}..."
  gunzip -f "data/${FILE_NAME}"
done
