from typing import Iterator, Tuple, Any

import os
import glob
import pickle
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image

# We assume a fixed language instruction here -- if your dataset has various instructions, please modify
LANGUAGE_INSTRUCTION = 'Do something'

# (180, 320) is the default resolution, modify if different resolution is desired
IMAGE_RES = (180, 320)


class ExampleDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # 定义数据集的信息结构，包括样本字段、类型、形状、说明等
    def _info(self) -> tfds.core.DatasetInfo:

        """Dataset metadata (homepage, citation,...)."""

        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
            'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'exterior_image_1_left': tfds.features.Image(
                            shape=(*IMAGE_RES, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Exterior camera 1 left viewpoint',
                        ),
                        'exterior_image_2_left': tfds.features.Image(
                            shape=(*IMAGE_RES, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Exterior camera 2 left viewpoint'
                        ),
                        'wrist_image_left': tfds.features.Image(
                            shape=(*IMAGE_RES, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Wrist camera RGB left viewpoint',
                        ),
                        'cartesian_position': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float64,
                            doc='Robot Cartesian state',
                        ),
                        'gripper_position': tfds.features.Tensor(
                            shape=(1,),
                            dtype=np.float64,
                            doc='Gripper position statae',
                        ),
                        'joint_position': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float64,
                            doc='Joint position state'
                        )
                    }),
                    'action_dict': tfds.features.FeaturesDict({
                        'cartesian_position': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float64,
                            doc='Commanded Cartesian position'
                        ),
                        'cartesian_velocity': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float64,
                            doc='Commanded Cartesian velocity'
                        ),
                        'gripper_position': tfds.features.Tensor(
                            shape=(1,),
                            dtype=np.float64,
                            doc='Commanded gripper position'
                        ),
                        'gripper_velocity': tfds.features.Tensor(
                            shape=(1,),
                            dtype=np.float64,
                            doc='Commanded gripper velocity'
                        ),
                        'joint_position': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float64,
                            doc='Commanded joint position'
                        ),
                        'joint_velocity': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float64,
                            doc='Commanded joint velocity'
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float64,
                        doc='Robot action, consists of [6x joint velocities, \
                            1x gripper position].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                    'recording_folderpath': tfds.features.Text(
                        doc='Path to the folder of recordings.'
                    )
                }),
            }))

    # 定义数据集的划分，如训练集与验证集
    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {
            'train': self._generate_examples('data/train'),
            'val': self._generate_examples('data/val'),
        }


    def _generate_examples(self, base_path) -> Iterator[Tuple[str, Any]]:
        """
        base_path: e.g. 'data/train' 或 'data/val'，里面包含多个 episode 文件夹
        """

        def _load_image(image_path):

            with Image.open(image_path) as img:

                # 更安全
                img = img.convert('RGB')
                return np.array(img)

        def _parse_episode(episode_dir):

            # episode_dir = 'data/train/episode1'
            pkl_files = sorted(glob.glob(os.path.join(episode_dir, '*.pkl')))
            image_dir = os.path.join(episode_dir, 'image')
            wrist_image_dir = os.path.join(episode_dir, 'wrist_image')

            episode = []
            for i, pkl_file in enumerate(pkl_files):

                basename = os.path.basename(pkl_file)
                timestamp = os.path.splitext(basename)[0]

                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)

                image_path = os.path.join(image_dir, f'left_{timestamp}.png')
                wrist_image_path = os.path.join(wrist_image_dir, f'wrist_left_{timestamp}.png')

                image = _load_image(image_path)
                wrist_image = _load_image(wrist_image_path)

                height, width, channels = image.shape

                fake_image = np.zeros((height, width, channels), dtype=image.dtype)

                fake_array_1 = np.zeros((1,), dtype=np.float64)
                fake_array_6 = np.zeros((6,), dtype=np.float64)
                fake_array_7 = np.zeros((7,), dtype=np.float64)

                # TODO: 疑问？为什么要在这里转 BGR?
                # [..., ::-1]
                episode.append({
                    'observation': {
                        'exterior_image_1_left': image[..., ::-1],  # 转BGR
                        'exterior_image_2_left': fake_image[..., ::-1],
                        'wrist_image_left': wrist_image[..., ::-1],
                        'cartesian_position': fake_array_6,
                        'joint_position': fake_array_7,
                        'gripper_position': fake_array_1,
                    },
                    'action_dict': {
                        'cartesian_position': fake_array_6,
                        'cartesian_velocity': fake_array_6,
                        'gripper_position': data["gripper_position"],
                        'gripper_velocity': fake_array_1,
                        'joint_position': data['joint_positions'],
                        'joint_velocity': data['joint_velocities'],
                    },
                    # TODO:
                    # 这个地方可能有问题，要单独测试一下 shape 是否是 （7，）
                    'action': np.concatenate([
                        data['joint_velocities'][:6],
                        np.array([data['joint_velocities'][0]])
                    ]),
                    'discount': 1.0,
                    'reward': float(i == (len(data) - 1)),
                    'is_first': i == 0,
                    'is_last': i == (len(data) - 1),
                    'is_terminal': i == (len(data) - 1),
                    'language_instruction': LANGUAGE_INSTRUCTION,
                })

            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_dir,
                }
            }

            return episode_dir, sample

        # 遍历 base_path 下所有 episode 文件夹
        # 'data/train/*'
        search_pattern = os.path.join(base_path, '*')

        # 匹配所有文件和文件夹
        all_files_and_dirs = glob.glob(search_pattern)
        """
        [
            'data/train/episode1',
            'data/train/episode2',
            'data/train/README.txt',
            'data/train/notes.docx'
        ]

        """

        # 只保留文件夹
        only_dirs = [path for path in all_files_and_dirs if os.path.isdir(path)]
        episode_dirs = sorted(only_dirs)
        """
            'data/train/episode1',
            'data/train/episode2',
        """

        for episode_dir in episode_dirs:
            yield _parse_episode(episode_dir)



        ############################################################


        ############################################################

        # 中文：（可选）大数据量可使用 Apache Beam 实现并行加载
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )
