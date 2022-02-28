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

import apache_beam as beam
from apache_beam.ml.inference.model_spec import ModelSpec
# TODO: import RunInferenceDoFn


def _unbatch(maybe_keyed_batches):
  keys, results = maybe_keyed_batches
  if keys:
    return zip(keys, results)
  else:
    return results


class RunInference(beam.PTransform):
  def __init__(self, model: ModelSpec, batch_size=None, **kwargs):
    self._model = model
    self._batch_size = batch_size

  def expand(self, pcoll: beam.PCollection) -> beam.PCollection:
    return (
        pcoll
        # TODO: Hook into the batching DoFn APIs.
        | beam.BatchElements(**self._batch_params)
        | beam.ParDo(RunInferenceDoFn(self._model))
        | beam.FlatMap(_unbatch))
