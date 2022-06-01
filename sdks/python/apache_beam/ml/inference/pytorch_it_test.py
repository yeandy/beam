#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# pylint: skip-file

"""End-to-End test for Pytorch Inference"""

import logging
import os
import unittest
import uuid

import pytest

import pandas as pd
from apache_beam.io.filesystems import FileSystems
from apache_beam.testing.test_pipeline import TestPipeline

try:
  import torch
  from apache_beam.examples.inference import pytorch_image_classification
  from apache_beam.examples.inference import pytorch_image_segmentation
except ImportError as e:
  torch = None

_EXPECTED_OUTPUTS = {
    'gs://apache-beam-ml/datasets/imagenet/raw-data/validation/ILSVRC2012_val_00005001.JPEG': '681',
    'gs://apache-beam-ml/datasets/imagenet/raw-data/validation/ILSVRC2012_val_00005002.JPEG': '333',
    'gs://apache-beam-ml/datasets/imagenet/raw-data/validation/ILSVRC2012_val_00005003.JPEG': '711',
    'gs://apache-beam-ml/datasets/imagenet/raw-data/validation/ILSVRC2012_val_00005004.JPEG': '286',
    'gs://apache-beam-ml/datasets/imagenet/raw-data/validation/ILSVRC2012_val_00005005.JPEG': '433',
    'gs://apache-beam-ml/datasets/imagenet/raw-data/validation/ILSVRC2012_val_00005006.JPEG': '290',
    'gs://apache-beam-ml/datasets/imagenet/raw-data/validation/ILSVRC2012_val_00005007.JPEG': '890',
    'gs://apache-beam-ml/datasets/imagenet/raw-data/validation/ILSVRC2012_val_00005008.JPEG': '592',
    'gs://apache-beam-ml/datasets/imagenet/raw-data/validation/ILSVRC2012_val_00005009.JPEG': '406',
    'gs://apache-beam-ml/datasets/imagenet/raw-data/validation/ILSVRC2012_val_00005010.JPEG': '996',
    'gs://apache-beam-ml/datasets/imagenet/raw-data/validation/ILSVRC2012_val_00005011.JPEG': '327',
    'gs://apache-beam-ml/datasets/imagenet/raw-data/validation/ILSVRC2012_val_00005012.JPEG': '573'
}


def process_outputs(filepath):
  with FileSystems().open(filepath) as f:
    lines = f.readlines()
  lines = [l.decode('utf-8').strip('\n') for l in lines]
  return lines


@unittest.skipIf(
    os.getenv('FORCE_TORCH_IT') is None and torch is None,
    'Missing dependencies. '
    'Test depends on torch, torchvision and pillow')
class PyTorchInference(unittest.TestCase):
  @pytest.mark.uses_pytorch
  @pytest.mark.it_postcommit
  def test_predictions_output_file(self):
    test_pipeline = TestPipeline(is_integration_test=True)
    # text files containing absolute path to the imagenet validation data on GCS
    file_of_image_names = 'gs://apache-beam-ml/testing/it_mobilenetv2_imagenet_validation_inputs.txt'  # disable: line-too-long
    output_file_dir = 'gs://apache-beam-ml/outputs/predictions'
    output_file = '/'.join([output_file_dir, str(uuid.uuid4()), 'result.txt'])

    model_state_dict_path = 'gs://apache-beam-ml/models/imagenet_classification_mobilenet_v2.pt'
    extra_opts = {
        'input': file_of_image_names,
        'output': output_file,
        'model_state_dict_path': model_state_dict_path,
    }
    pytorch_image_classification.run(
        test_pipeline.get_full_options_as_args(**extra_opts),
        save_main_session=False)

    self.assertEqual(FileSystems().exists(output_file), True)
    predictions = process_outputs(filepath=output_file)

    for prediction in predictions:
      filename, prediction = prediction.split(',')
      self.assertEqual(_EXPECTED_OUTPUTS[filename], prediction)

  @pytest.mark.uses_pytorch
  @pytest.mark.inference_benchmark
  def test_torch_imagenet_benchmark(self):
    test_pipeline = TestPipeline(is_integration_test=True)
    # text files containing absolute path to the imagenet validation data on GCS
    file_of_image_names = 'gs://apache-beam-ml/testing/imagenet_validation_inputs.txt'
    output_file_dir = 'gs://apache-beam-ml/outputs'
    output_file = '/'.join([output_file_dir, str(uuid.uuid4()), 'result.txt'])

    model_state_dict_path = 'gs://apache-beam-ml/models/imagenet_classification_mobilenet_v2.pt'
    extra_opts = {
        'input': file_of_image_names,
        'output': output_file,
        'model_state_dict_path': model_state_dict_path,
    }
    pytorch_image_classification.run(
        test_pipeline.get_full_options_as_args(**extra_opts),
        save_main_session=False)

    self.assertEqual(FileSystems().exists(output_file), True)
    predictions = process_outputs(filepath=output_file)

  @pytest.mark.it_postcommit
  @pytest.mark.uses_pytorch
  def test_image_segmentation_small(self):
    test_pipeline = TestPipeline(is_integration_test=True)
    output_file_dir = 'gs://apache-beam-ml/testing'
    output_file = '/'.join(
        [output_file_dir, 'predictions', str(uuid.uuid4()), 'result.txt'])
    actuals_file = '/'.join([
        output_file_dir,
        'expected_outputs',
        'torchvision_maskrcnn_resnet50_actuals.txt'
    ])
    file_of_image_names = 'gs://apache-beam-ml/datasets/coco/raw-data/annotations/instances_val2017.json'
    base_output_files_dir = 'gs://apache-beam-ml/datasets/coco/raw-data/val2017'
    filtered_image_ids = [
        397133,
        37777,
        252219,
        87038,
        174482,
        403385,
        6818,
        480985,
        458054,
        331352,
        296649,
        386912,
        502136,
        491497,
        184791
    ]

    # State dict is obtained by saving the maskrcnn_resnet50_fpn(pretrained=True) model
    model_path = 'gs://apache-beam-ml/models/torchvision.models.detection.maskrcnn_resnet50_fpn.pth'
    extra_opts = {
        'input': file_of_image_names,
        'output': output_file,
        'model_path': model_path,
        'images_dir': base_output_files_dir,
        'filtered_image_ids': ','.join((str(id) for id in filtered_image_ids))
    }

    pytorch_image_segmentation.run(
        test_pipeline.get_full_options_as_args(**extra_opts),
        save_main_session=False)

    self.assertEqual(FileSystems().exists(output_file), True)
    predictions = process_outputs(filepath=output_file)
    actuals = process_outputs(filepath=actuals_file)

    predictions_dict = {}
    for prediction in predictions:
      id, prediction_labels = prediction.split(';')
      predictions_dict[id] = prediction_labels

    for actual in actuals:
      image_id, actual_labels = actual.split(';')
      prediction_labels = predictions_dict[image_id]
      self.assertEqual(actual_labels, prediction_labels)

  @pytest.mark.inference_benchmark
  @pytest.mark.no_xdist
  @pytest.mark.timeout(2000)
  @pytest.mark.uses_pytorch
  def test_image_segmentation_benchmark(self):
    test_pipeline = TestPipeline(is_integration_test=True)
    file_of_image_names = 'gs://apache-beam-ml/datasets/coco/raw-data/annotations/instances_val2017.json'
    base_output_files_dir = 'gs://apache-beam-ml/datasets/coco/raw-data/val2017'

    # State dict is obtained by saving the maskrcnn_resnet50_fpn(pretrained=True) model
    model_path = 'gs://apache-beam-ml/models/torchvision.models.detection.maskrcnn_resnet50_fpn.pth'
    extra_opts = {
        'input': file_of_image_names,
        'model_path': model_path,
        'images_dir': base_output_files_dir,
        'n_examples': 5000
    }

    pytorch_image_segmentation.run(
        test_pipeline.get_full_options_as_args(**extra_opts),
        save_main_session=False)


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.DEBUG)
  unittest.main()
