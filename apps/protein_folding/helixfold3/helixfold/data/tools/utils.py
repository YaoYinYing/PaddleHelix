#   Copyright (c) 2024 PaddleHelix Authors. All Rights Reserved.
#
# Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0
# International License (the "License");  you may not use this file  except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common utilities for data pipeline tools."""

import contextlib
import shutil
from absl import logging
import tempfile
import time
from typing import Optional


@contextlib.contextmanager
def tmpdir_manager(base_dir: Optional[str] = None):
  """Context manager that deletes a temporary directory on exit."""
  tmpdir = tempfile.mkdtemp(dir=base_dir)
  try:
    yield tmpdir
  finally:
    shutil.rmtree(tmpdir, ignore_errors=True)


@contextlib.contextmanager
def timing(msg: str):
  logging.info('Started %s', msg)
  tic = time.time()
  yield
  toc = time.time()
  logging.info('Finished %s in %.3f seconds', msg, toc - tic)



def minibatches(inputs_data, batch_size):
  """
  Generates minibatches from input data with a specified batch size.

  Args:
  - inputs_data (list or iterable): Input data to be divided into minibatches.
  - batch_size (int): Size of each minibatch.

  Yields:
  - list: Minibatches of data based on the specified batch size.

  Note:
  If the length of the inputs_data is not perfectly divisible by the batch_size,
  the last batch may have fewer elements.

  Example Usage:
  ```python
  data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  batch_size = 3
  for batch in minibatches(data, batch_size):
      print(batch)
  ```
  """
  for start_idx in range(0, len(inputs_data), batch_size):
    if len(inputs_data[start_idx:]) > batch_size:
      excerpt = slice(start_idx, start_idx + batch_size)
      logging.info("Send data in length: %s" % len(inputs_data[excerpt]))
      yield inputs_data[excerpt]
    else:
      logging.info(
          "Send final data in length: %s" % len(inputs_data[start_idx:])
      )
      yield inputs_data[start_idx:]
