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

# pytype: skip-file

from typing import Iterable
from typing import List
from typing import Union

import numpy as np
import pickle
import torch
from torch import nn

from apache_beam.ml.inference.base import InferenceRunner
from apache_beam.ml.inference.base import ModelLoader


class PytorchInferenceRunner(InferenceRunner):
  """
  Implements Pytorch inference method
  """
  def __init__(self, input_dim: int, device: torch.device):
    self._input_dim = input_dim
    self._device = device

  def run_inference(
      self, batch: List[Union[np.ndarray, torch.Tensor]],
      model: nn.Module) -> Iterable[torch.Tensor]:
    """
    Runs inferences on a batch of examples and returns an Iterable of
    Predictions."""
    if batch:
      if isinstance(batch[0], np.ndarray):
        batch = torch.Tensor(batch)
      elif isinstance(batch[0], torch.Tensor):
        batch = torch.stack(batch)
      else:
        raise ValueError("PCollection must be an numpy array or a torch Tensor")

    if batch.device != self._device:
      batch = batch.to(self._device)
    return model(batch)

  def get_num_bytes(self, batch: List[torch.Tensor]) -> int:
    """Returns the number of bytes of data for a batch."""
    total_size = 0
    for el in batch:
      if isinstance(el, np.ndarray):
        total_size += el.itemsize
      elif isinstance(el, torch.Tensor):
        total_size += el.element_size()
      else:
        total_size += len(pickle.dumps(el))
    return total_size

  def get_metrics_namespace(self) -> str:
    """
    Returns a namespace for metrics collected by the RunInference transform.
    """
    return 'RunInferencePytorch'


class PytorchModelLoader(ModelLoader):
  """Loads a Pytorch Model."""
  def __init__(
      self,
      input_dim: int,
      state_dict_path: str,
      model_class: nn.Module,
      device: str = 'CPU',
      map_location=None):
    """
    input_dim: dimension (# of features) of the data
    state_dict_path: path to the saved dictionary of the model state.
    model_class: class of the Pytorch model that defines the model structure.
    device: the device on which you wish to run the model. If ``device = GPU``
        then device will be cuda if it is avaiable. Otherwise, it will be cpu.
    map_location:

    See https://pytorch.org/tutorials/beginner/saving_loading_models.html
    for details
    """
    self._input_dim = input_dim
    self._state_dict_path = state_dict_path
    if device == 'GPU' and torch.cuda.is_available():
      self._device = torch.device('cuda')
    else:
      self._device = torch.device('cpu')
    self._model_class = model_class
    self._model_class.to(self._device)
    self._map_location = map_location

  def load_model(self) -> nn.Module:
    """Loads and initializes a Pytorch model for processing."""
    model = self._model_class
    model.load_state_dict(
        torch.load(self._state_dict_path, map_location=self._map_location))
    model.eval()
    return model

  def get_inference_runner(self) -> InferenceRunner:
    """Returns a Pytorch implementation of InferenceRunner."""
    return PytorchInferenceRunner(
        input_dim=self._input_dim, device=self._device)
