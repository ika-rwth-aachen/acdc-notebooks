"""evilog_2021 dataset."""

import resource
low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

import tensorflow as tf
import tensorflow_datasets as tfds

import os
from pyntcloud import PyntCloud
from PIL import Image
import numpy as np


_DESCRIPTION = """
The EviLOG 2021 dataset contains point clouds and evidential occupancy grid maps.
Point clouds have shape (NUM_POINTS, 4) where each points consits of (x, y, z, intensity).
Occupancy grid maps are of shape (HEIGHT, WIDTH, 2) where each cell contains the two belief masses m_occupied and m_free.

- **Synthetic training and validation data** consisting of lidar point clouds and evidential occupancy grid maps (as png files)
  - 10.000 training samples
  - 1.000 validation samples
  - 100 test samples
- **Real-world input data** that was recorded with a Velodyne VLP32C lidar sensor during a ~9 minutes ride in an urban area (5.224 point clouds).
"""

_CITATION = """
@misc{vankempen2021simulationbased,
      title={A Simulation-based End-to-End Learning Framework for Evidential Occupancy Grid Mapping}, 
      author={Raphael van Kempen, Bastian Lampe, Timo Woopen and Lutz Eckstein},
      year={2021},
      eprint={2102.12718},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
"""

VERSION = tfds.core.Version('1.0.0')

RELEASE_NOTES = {
  '1.0.0': 'Initial release.',
}


class Evilog2021Config(tfds.core.BuilderConfig):
  """BuilderConfig for Evilog2021."""

  def __init__(self, *, variant=None, **kwargs):
    """BuilderConfig for Evilog2021.
    Args:
      variant: str. Variant of the dataset.
      **kwargs: keyword arguments forwarded to super.
    """
    super(Evilog2021Config, self).__init__(version=VERSION, **kwargs)
    self.variant = variant


class Evilog2021(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for evilog_2021 dataset."""

  BUILDER_CONFIGS = [
    Evilog2021Config(
      name="full",
      description="Full Dataset",
      variant="full",
    ),
    Evilog2021Config(
      name="demo",
      description="Demo Dataset",
      variant="demo",
    ),
  ]

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'point_cloud': tfds.features.Tensor(shape=(None, 4), dtype=tf.float32),
            'grid_map': tfds.features.Tensor(shape=(None, None, 2), dtype=tf.float32, encoding='zlib'),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('point_cloud', 'grid_map'),  # Set to `None` to disable
        homepage='https://github.com/ika-rwth-aachen/EviLOG',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    if self._builder_config.variant == "demo":
      url = 'https://rwth-aachen.sciebo.de/s/tnvuKHcIzPAA4QK/download'
    else:
      url = '******'

    path = dl_manager.download_and_extract(url)
    print(path)

    return {
        'train': self._generate_examples(
            point_cloud_path = path / 'input_train',
            grid_map_path = path / 'label_train'
        ),
        'valid': self._generate_examples(
            point_cloud_path = path / 'input_valid',
            grid_map_path = path / 'label_valid'
        ),
        'test': self._generate_examples(
            point_cloud_path = path / 'input_test',
            grid_map_path = path / 'label_test'
        ),
        'real': self._generate_examples(
            point_cloud_path = path / 'input_real'
        ),
    }

  def _generate_examples(self, point_cloud_path, grid_map_path=None):
    """Yields examples."""
    for input_file in point_cloud_path.glob('*.pcd'):
      filename = os.path.splitext(os.path.basename(input_file))[0]
    
      # convert pcd file to numpy.ndarray with one point per row with columns (x, y, z, i)
      point_cloud = PyntCloud.from_file(str(input_file)).points.values[:, 0:4]
      
      if grid_map_path is None:
        grid_map = np.array([[[0, 0]]], dtype=np.float32)
      else:
        # convert png file to grid map of size [height, width, 2] with (m_occupied, m_free) in each cell
        label_file = os.path.join(grid_map_path, filename + '.png')
        image = Image.open(label_file)
        grid_map = np.asarray(image, dtype=np.float32)
        grid_map = grid_map[..., 1:3]/255.0  # use only channels 'free' and 'occupied'

      yield filename, {
          'point_cloud': point_cloud,
          'grid_map': grid_map,
      }

