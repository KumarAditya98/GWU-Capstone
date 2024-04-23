#!/bin/bash

if [ ! -d dataset ]; then
  mkdir dataset
fi
cd dataset
mkdir medical_data
cd medical_data

echo "Downloading and unzipping dataset"
wget https://zenodo.org/records/10499039/files/ImageClef-2019-VQA-Med-Training.zip?download=1
wget https://zenodo.org/records/10499039/files/ImageClef-2019-VQA-Med-Validation.zip?download=1
wget https://zenodo.org/records/10499039/files/VQAMed2019Test.zip?download=1
unzip '*.zip?download=1'
rm *.zip\?download\=1
mv ImageClef-2019-VQA-Med-Training train
mv ImageClef-2019-VQA-Med-Validation val
mv VQAMed2019Test test
wget https://files.osf.io/v1/resources/89kps/providers/osfstorage/?zip=
unzip  'index.html?zip='
rm 'VQA_RAD Dataset Public.xlsx'
rm 'VQA_RAD Dataset Public.xml'
rm index.html?zip=
cd test
unzip '*.zip'
rm *.zip
pip install ruamel_yaml
pip install rouge_score
pip install imgaug
pip install torchmetrics