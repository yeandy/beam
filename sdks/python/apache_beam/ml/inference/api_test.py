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

import unittest

import pytest  # pylint: disable=unused-import

from apache_beam.ml.inference.api import PyTorchDevice
from apache_beam.ml.inference.api import PyTorchModelSpec
from apache_beam.ml.inference.api import SklearnModelSpec
from apache_beam.ml.inference.api import SklearnSerializationType


class RunInferenceTest(unittest.TestCase):
  def test_valid_pytorch_model(self):
    model = PyTorchModelSpec(
        model_uri='pytorch_model.pth', device=PyTorchDevice.GPU)
    self.assertEqual(model.model_uri, 'pytorch_model.pth')
    self.assertEqual(model.device, PyTorchDevice.GPU)

  def test_invalid_pytorch_model(self):
    with pytest.raises(TypeError, match='device'):
      PyTorchModelSpec(model_uri='pytorch_model.pth')
    with pytest.raises(TypeError, match='model_uri'):
      PyTorchModelSpec(device=PyTorchDevice.CPU)

  def test_valid_sklearn_model(self):
    model = SklearnModelSpec(
        model_uri='sklearn_model.pickle',
        serialization_type=SklearnSerializationType.PICKLE)
    self.assertEqual(model.model_uri, 'sklearn_model.pickle')
    self.assertEqual(model.serialization_type, SklearnSerializationType.PICKLE)

  def test_invalid_sklearn_model(self):
    with pytest.raises(TypeError, match='serialization_type'):
      SklearnModelSpec(model_uri='sklearn_model.pickle')
    with pytest.raises(TypeError, match='model_uri'):
      SklearnModelSpec(serialization_type=SklearnSerializationType.PICKLE)


if __name__ == '__main__':
  unittest.main()
