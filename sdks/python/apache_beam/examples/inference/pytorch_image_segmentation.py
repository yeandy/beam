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

import argparse
import io
import json
import os
from functools import partial
from typing import Any
from typing import Iterable
from typing import Tuple
from typing import Union

import apache_beam as beam
import torch
import torchvision
from apache_beam.io.filesystems import FileSystems
from apache_beam.ml.inference.api import PredictionResult
from apache_beam.ml.inference.api import RunInference
from apache_beam.ml.inference.pytorch import PytorchModelLoader
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
from PIL import Image
from torchvision import transforms


def read_image(image_id_name_tuple: Tuple[int, str],
               path_to_dir: str = None) -> Tuple[int, str, Image.Image]:
  image_id, image_file_name = image_id_name_tuple
  image_file_name = os.path.join(path_to_dir, image_file_name)
  with FileSystems().open(image_file_name, 'r') as file:
    data = Image.open(io.BytesIO(file.read())).convert('RGB')
    return image_id, image_file_name, data


def preprocess_data(data: Image) -> torch.Tensor:
  image_size = (800, 800)
  normalize = transforms.Normalize(
      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  transform = transforms.Compose([
      transforms.Resize(image_size),
      transforms.ToTensor(),
      normalize,
  ])
  return transform(data)


def split_json(elements):
  for element in elements:
    yield element


class PostProcessor(beam.DoFn):
  "Output filename and prediction"

  def process(
      self, element: Union[PredictionResult, Tuple[Any, PredictionResult]]
  ) -> Iterable[str]:
    image_id, prediction_result = element
    labels = prediction_result.inference['labels']
    yield str(image_id) + ';' + str(labels.tolist())


def run_pipeline(options: PipelineOptions, args=None):
  """Sets up PyTorch RunInference pipeline"""
  # model_class = torchvision.models.efficientnet_b0
  model_class = torchvision.models.detection.maskrcnn_resnet50_fpn
  model_params = {'pretrained': False}
  model_loader = PytorchModelLoader(
      state_dict_path=args.model_path,
      model_class=model_class,
      model_params=model_params)
  with beam.Pipeline(options=options) as p:
    # Read in json
    input_json = (
        p
        | 'Read from json file' >> beam.io.ReadFromText(args.input)
        | "Load json" >> beam.Map(json.loads))

    # Get image examples
    images_json = (
        input_json
        | 'Enter images json' >> beam.Map(lambda json: json['images'])
        | 'Split image list' >> beam.FlatMap(split_json)
        # FlatMap is a high fan-out operation. Fusion limits parallelism.
        # Reshuffle prevents fusion
        # https://cloud.google.com/dataflow/docs/guides/deploying-a-pipeline#preventing-fusion
        | 'Reshuffle images' >> beam.Reshuffle())
    image_names = (
        images_json
        |
        'Extract image names' >> beam.Map(lambda x: (x['id'], x['file_name'])))
    if args.filtered_image_ids:
      image_names = (
          image_names
          | 'filter images' >> beam.Filter(
              lambda x: str(x[0]) in args.filtered_image_ids.split(',')))
    images = (
        image_names
        | 'Parse and read files from the input_file' >> beam.Map(
            partial(read_image, path_to_dir=args.images_dir))
        | 'Preprocess images' >> beam.MapTuple(
            lambda image_id, _, data: (image_id, preprocess_data(data))))

    # Get ground truth labels
    annotations_json = (
        input_json
        |
        'Enter annotations json' >> beam.Map(lambda json: json['annotations'])
        | 'Split annotations list' >> beam.FlatMap(split_json)
        | 'Reshuffle annotations' >> beam.Reshuffle())
    category_ids = (
        annotations_json
        | 'Extract category ids' >>
        beam.Map(lambda x: (x['image_id'], x['category_id'])))
    if args.filtered_image_ids:
      category_ids = (
          category_ids
          | 'filter categories' >> beam.Filter(
              lambda x: str(x[0]) in args.filtered_image_ids.split(',')))

    # Call RunInference
    predictions = (
        images
        | 'PyTorch RunInference' >> RunInference(model_loader))

    # Writing to a file for small IT tests
    if args.output:
      post_processed_data = (
          predictions
          | 'Process output' >> beam.ParDo(PostProcessor()))
      post_processed_data | "Write output to GCS" >> beam.io.WriteToText( # pylint: disable=expression-not-assigned
        args.output,
        shard_name_template='',
        append_trailing_newlines=True)
    else:
      # Don't write to GCS for large benchmark tests. Merge with actuals and
      # confirm counts
      post_processed_data = (
          predictions
          | 'Extract inferences' >> beam.Map(lambda x: (x[0], x[1].inference)))
      merged = (({
          'actual_labels': category_ids, 'predictions': post_processed_data
      })
                | 'Merge' >> beam.CoGroupByKey())

      count = (
          merged
          | 'Check if both input and predictions are non empty' >> beam.Map(
              lambda x: int((
                  len(x[1]['actual_labels']) >= 0 and len(x[1]['predictions']) >
                  0)))
          | 'Sum values' >> beam.CombineGlobally(sum))
      assert_that(count, equal_to([int(args.n_examples)]))


def parse_known_args(argv):
  """Parses args for the workflow."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--input',
      dest='input',
      required=True,
      help='Path to the CSV file containing image names')
  parser.add_argument(
      '--output',
      dest='output',
      help='Predictions are saved to the output'
      ' text file.')
  parser.add_argument(
      '--model_path', dest='model_path', help='Path to load the model.')
  parser.add_argument(
      '--images_dir', required=True, help='Path to the images folder')
  parser.add_argument(
      '--filtered_image_ids',
      required=False,
      help='List of image ids to filter')
  parser.add_argument(
      '--n_examples',
      required=False,
      help='Expected number of examples and corresponding predictions')
  return parser.parse_known_args(argv)


def run(argv=None, save_main_session=True):
  """Entry point. Defines and runs the pipeline."""
  known_args, pipeline_args = parse_known_args(argv)
  pipeline_options = PipelineOptions(flags=pipeline_args)
  pipeline_options.view_as(SetupOptions).save_main_session = save_main_session
  run_pipeline(pipeline_options, args=known_args)


if __name__ == '__main__':
  run()
