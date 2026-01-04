#!/bin/bash

# Copyright (c) 2022 Hongji Wang (jijijiang77@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

download_dir=/path/to/dataset

[ ! -d ${download_dir} ] && mkdir -p ${download_dir}

if [ ! -f ${download_dir}/vox1_test_wav.zip ]; then
  echo "Downloading vox1_test_wav.zip ..."
  wget --user USERNAME --password='PASSWORD' https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_test_wav.zip -P ${download_dir}
  md5=$(md5sum ${download_dir}/vox1_test_wav.zip | awk '{print $1}')
  [ $md5 != "185fdc63c3c739954633d50379a3d102" ] && echo "Wrong md5sum of vox1_test_wav.zip" && exit 1
fi

if [ ! -f ${download_dir}/vox1_dev_wav.zip ]; then
  echo "Downloading vox1_dev_wav.zip ..."
  for part in a b c d; do
    wget --user USERNAME --password='PASSWORD' https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_parta${part} -P ${download_dir} &
  done
  wait
  cat ${download_dir}/vox1_dev* >${download_dir}/vox1_dev_wav.zip
  md5=$(md5sum ${download_dir}/vox1_dev_wav.zip | awk '{print $1}')
  [ $md5 != "ae63e55b951748cc486645f532ba230b" ] && echo "Wrong md5sum of vox1_dev_wav.zip" && exit 1
fi

echo "Download success !!!"