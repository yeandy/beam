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
from apache_beam.ml.inference.api import PyTorchModel
from apache_beam.ml.inference.api import SklearnModel
import pytest  # pylint: disable=unused-import


class RunInferenceTest(unittest.TestCase):
  def test_valid_pytorch_model(self):
    model = PyTorchModel(model_url='pytorch_model.pth', device='gpu')
    assert model.model_url == 'pytorch_model.pth'
    assert model.device == 'GPU'

  def test_invalid_pytorch_model(self):
    with pytest.raises(TypeError, match='device'):
      PyTorchModel(model_url='pytorch_model.pth')
    with pytest.raises(TypeError, match='model_url'):
      PyTorchModel(device='cpu')

  def test_valid_sklearn_model(self):
    model = SklearnModel(
        model_url='sklearn_model.pickle', serialization_type='Pickle')
    assert model.model_url == 'sklearn_model.pickle'
    assert model.serialization_type == 'PICKLE'

  def test_invalid_sklearn_model(self):
    with pytest.raises(TypeError, match='serialization_type'):
      SklearnModel(model_url='sklearn_model.pickle')
    with pytest.raises(TypeError, match='model_url'):
      SklearnModel(serialization_type='Pickle')


if __name__ == '__main__':
  unittest.main()
