
import pickle
import os
from pathlib import Path
from datetime import datetime
import re
from PIL import Image
import numpy as np
from tqdm import tqdm
import tyro

"""
在运行这个转换脚本之前，我们假设：
1.图像已经写入了: bc_data/gello/XXXXXX/image 文件夹, 并且以时间戳命名
2.pkl 文件已经写入了: bc_data/gello/XXXXXX/ 文件夹
3.main(data_dir): data_dir 是所有数据保存的文件夹，在这个示例中是 bc_data/gello, 再往下是每个episode的文件夹
4.传入参数 data_dir, prompt。
5.data_dir 在3.已经解释过了, prompt: str，表示data_dir这整个文件夹下所有任务的内容，也就是每个 episode 干了啥
"""

from lerobot.src.lerobot.datasets.lerobot_dataset import LeRobotDataset

def main(data_dir : str, prompt : str):
    """
    参数:
        data_dir : 数据保存的文件夹名称，例如 bc_data/gello
        prompt: 任务的 prompt
    """

    dataset = LeRobotDataset.create(
            repo_id="REPO_PI_FAST_DATASETS",
            robot_type="panda",
            fps=10,
            features={
                "image": {
                    "dtype": "image",
                    "shape": (720, 1080, 3),
                    "names": ["height", "width", "channel"],
                },
                "wrist_image": {
                    "dtype": "image",
                    "shape": (720, 1080, 3),
                    "names": ["height", "width", "channel"],
                },
                "state": {
                    "dtype": "float64",
                    "shape": (8,), # 7 - Dof + 1 个夹爪
                    "names": ["state"],
                },
                "actions": {
                    "dtype": "float64",
                    "shape": (8,),
                    "names": ["actions"],
                },
            },
            image_writer_threads=10,
            image_writer_processes=5,
        )

    # data_dir = "./bc_data/gello"
    for episode_name in os.listdir(data_dir):

        image_path = os.path.join(data_dir, episode_name, "image")
        wrist_image_path = os.path.join(data_dir, episode_name, "wrist_image")
        
        pkl_files = list(os.path.join(data_dir, episode_name).glob('*.pkl'))

        pkl_sorted_files = sorted(pkl_files)
        image_files = sorted(image_files)
        wrist_image_files = sorted(wrist_image_files)

        for idx, image_file in enumerate(tqdm(image_files, desc="正在进行数据转换")):
            image = Image.open(os.path.join(image_path, image_file))
            image_np = np.array(image)
            wrist_image = Image.open(os.path.join(wrist_image_path, image_file))
            wrist_image_np = np.array(wrist_image)

            assert image_np.shape[2] == 3, "主视角图像格式不匹配"
            assert wrist_image_np.shape[2] == 3, "腕部视角图像格式不匹配"

            try:
                with open(os.path.join(data_dir, episode_name, pkl_sorted_files[idx]), "rb") as f:
                    data = pickle.load(f)
            except Exception as e:
                print(e)
            
            """
            {
                'joint_positions': np.ndarray,
                'joint_velocities': np.ndarray,
                'ee_pos_quat': np.ndarray,
                'gripper_position': np.float32,
                'control': np.ndarray
            }
            """
            # image_np, wrist_image_np 应该是 0-255
            dataset.add_frame({
                "image": image_np,
                "wrist_image": wrist_image_np,
                "state": data["joint_positions"],
                "actions": data["control"],
            })

        dataset.save_episode(task=prompt) 

    dataset.consolidate(run_compute_stats=False) 

if __name__ == "__main__":
    # 指定 data_dir, prompt,
    main(tyro.cli())
