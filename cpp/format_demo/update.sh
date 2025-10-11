#!/bin/sh

cp feature_processor.* /home/manu/nfs/VSCodeProject/smartboxcore/app/algorithm/ -rvf
cp cores/face/FaceAnalyzer.* /home/manu/nfs/VSCodeProject/smartboxcore/app/algorithm/cores/face/ -rvf
cp cores/personReid/PersonReid_dla.* /home/manu/nfs/VSCodeProject/smartboxcore/app/algorithm/cores/personReid/ -rvf

echo "Done !!!"