from typing import Any, Dict
import numpy as np
from PIL import Image

################################################################################################
#                                        目标配置                                              #
################################################################################################
# 以下是目标数据集的特征配置，定义了RLDS格式中每个步骤（step）的结构
# features=tfds.features.FeaturesDict({
#     'steps': tfds.features.Dataset({  # 每个步骤的数据集
#         'observation': tfds.features.FeaturesDict({  # 观测数据
#             'image': tfds.features.Image(  # 图像特征
#                 shape=(128, 128, 3),  # 图像尺寸：128x128像素，3通道（RGB）
#                 dtype=np.uint8,  # 数据类型：无符号8位整数
#                 encoding_format='jpeg',  # 图像编码格式：JPEG
#                 doc='Main camera RGB observation.',  # 描述：主摄像头RGB观测
#             ),
#         }),
#         'action': tfds.features.Tensor(  # 动作数据
#             shape=(8,),  # 动作维度：8维
#             dtype=np.float32,  # 数据类型：32位浮点数
#             doc='Robot action, consists of [3x EEF position, '
#                 '3x EEF orientation yaw/pitch/roll, 1x gripper open/close position, '
#                 '1x terminate episode].',  # 描述：机器人动作，包括[3维末端执行器位置，3维末端执行器方向（偏航/俯仰/滚转），1维夹爪开/关，1维终止片段]
#         ),
#         'discount': tfds.features.Scalar(  # 折扣因子
#             dtype=np.float32,  # 数据类型：32位浮点数
#             doc='Discount if provided, default to 1.'  # 描述：折扣因子，默认值为1
#         ),
#         'reward': tfds.features.Scalar(  # 奖励值
#             dtype=np.float32,  # 数据类型：32位浮点数
#             doc='Reward if provided, 1 on final step for demos.'  # 描述：奖励值，在演示的最后一步为1
#         ),
#         'is_first': tfds.features.Scalar(  # 是否为片段的第一步
#             dtype=np.bool_,  # 数据类型：布尔值
#             doc='True on first step of the episode.'  # 描述：如果为片段的第一步，则为True
#         ),
#         'is_last': tfds.features.Scalar(  # 是否为片段的最后一步
#             dtype=np.bool_,  # 数据类型：布尔值
#             doc='True on last step of the episode.'  # 描述：如果为片段的最后一步，则为True
#         ),
#         'is_terminal': tfds.features.Scalar(  # 是否为终止步骤
#             dtype=np.bool_,  # 数据类型：布尔值
#             doc='True on last step of the episode if it is a terminal step, True for demos.'  # 描述：如果为片段的最后一步且为终止步骤，则为True，对演示为True
#         ),
#         'language_instruction': tfds.features.Text(  # 语言指令
#             doc='Language Instruction.'  # 描述：语言指令
#         ),
#         'language_embedding': tfds.features.Tensor(  # 语言嵌入
#             shape=(512,),  # 嵌入维度：512维
#             dtype=np.float32,  # 数据类型：32位浮点数
#             doc='Kona language embedding. '
#                 'See https://tfhub.dev/google/universal-sentence-encoder-large/5'  # 描述：Kona语言嵌入，参考链接
#         ),
#     })
################################################################################################
#                                                                                              #
################################################################################################

def transform_step(step: Dict[str, Any]) -> Dict[str, Any]:
    """将源数据集的步骤映射到目标数据集配置。
       输入是一个包含numpy数组的字典。

    参数:
        step (Dict[str, Any]): 源数据集中的一个步骤，包含观测、动作等字段

    返回:
        Dict[str, Any]: 转换后的步骤，符合目标数据集的配置
    """
    # 将输入步骤的观测图像从numpy数组转换为PIL图像，并调整大小为128x128，使用LANCZOS重采样方法
    img = Image.fromarray(step['observation']['image']).resize(
        (128, 128), Image.Resampling.LANCZOS)
    
    # 创建转换后的步骤字典，初始化观测和动作字段
    transformed_step = {
        'observation': {
            'image': np.array(img),  # 将调整大小后的PIL图像转换回numpy数组
        },
        # 动作字段：从源动作中提取前3维（末端执行器位置）、第6-8维（末端执行器方向）、最后2维（夹爪开/关和终止标志）
        'action': np.concatenate(
            [step['action'][:3], step['action'][5:8], step['action'][-2:]]),
    }

    # 复制其他字段，不做任何更改
    for copy_key in ['discount', 'reward', 'is_first', 'is_last', 'is_terminal',
                     'language_instruction', 'language_embedding']:
        transformed_step[copy_key] = step[copy_key]

    # 返回转换后的步骤
    return transformed_step