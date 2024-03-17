#!/bin/bash

mkdir dataset
cd dataset

mkdir vqa_v2
cd vqa_v2
echo "Downloading Coco dataset."
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/zips/test2015.zip

echo "Unzipping Coco dataset."
unzip '*.zip'

rm *.zip

cd ..

mkdir vg
cd vg
echo "Downloading Visual Genome dataset."
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip


echo "Unzipping Visual Genome dataset."
unzip '*.zip'

mkdir image

mv VG_100K/* image
mv VG_100K_2/* image
rmdir VG_100K
rmdir VG_100K_2

rm *.zip

cd ..

echo "Installing few dependencies."
pip install ruamel.yaml
pip install timm
pip install pycocotools
pip install fairscale

git clone https://github.com/salaniz/pycocoevalcap.git
cd pycocoevalcap
python3 setup.py build_ext install

pip install fairscale

cd ..

echo "All tasks completed successfully."
