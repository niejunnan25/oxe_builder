from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
# import tensorflow_hub as hub

class ExampleDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    # 初始化类时，加载 Universal Sentence Encoder 模型用于生成语言指令的嵌入
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    # 定义数据集的信息结构，包括样本字段、类型、形状、说明等
    def _info(self) -> tfds.core.DatasetInfo:

        """Dataset metadata (homepage, citation,...)."""

        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({

                        'image': tfds.features.Image(
                            shape=(64, 64, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),

                        'wrist_image': tfds.features.Image(
                            shape=(64, 64, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Wrist camera RGB observation.',
                        ),

                        'state': tfds.features.Tensor(
                            shape=(10,),
                            dtype=np.float32,
                            doc='Robot state, consists of [7x robot joint angles, '
                                '2x gripper position, 1x door opening angle].',
                        )
                    }),

                    'action': tfds.features.Tensor(
                        shape=(10,),
                        dtype=np.float32,
                        doc='Robot action, consists of [7x joint velocities, '
                            '2x gripper velocities, 1x terminate episode].',
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

                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),

                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    # 定义数据集的划分，如训练集与验证集
    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='data/train/episode_*.npy'),
            'val': self._generate_examples(path='data/val/episode_*.npy'),
        }

    # 数据生成函数，根据路径逐个生成样本
    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        # 解析一个 episode 文件，返回其内容
        def _parse_example(episode_path):

            # 加载 npy 文件，数据为包含多个字典的列表，每个字典表示一个 step
            data = np.load(episode_path, allow_pickle=True)  # this is a list of dicts in our case

            # 构建当前 episode 的所有 step
            episode = []
            for i, step in enumerate(data):
                # 中文：计算语言指令的嵌入向量（512维）
                language_embedding = self._embed([step['language_instruction']])[0].numpy()

                # 中文：添加当前 step 到 episode 中
                episode.append({
                    'observation': {
                        'image': step['image'],
                        'wrist_image': step['wrist_image'],
                        'state': step['state'],
                    },
                    'action': step['action'],
                    'discount': 1.0,
                    'reward': float(i == (len(data) - 1)),  # 中文：只有最后一个 step 的 reward 为 1
                    'is_first': i == 0,
                    'is_last': i == (len(data) - 1),
                    'is_terminal': i == (len(data) - 1),
                    'language_instruction': step['language_instruction'],
                    # 'language_embedding': language_embedding,
                })

            # 构造一个最终样本
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            # 返回 (键，样本) 对
            return episode_path, sample

        # 查找所有符合路径模式的 episode 文件
        episode_paths = glob.glob(path)

        # 顺序生成样本（适合中小数据量）
        for sample in episode_paths:
            yield _parse_example(sample)

        # 中文：（可选）大数据量可使用 Apache Beam 实现并行加载
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )
