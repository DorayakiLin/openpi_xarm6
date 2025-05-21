from pathlib import Path
import shutil

import cv2
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import pyarrow.parquet as pq
import tyro

REPO_NAME = "DorayakiLin_parquet_pick_bread_v1"  # 可改成你希望的输出子目录名


def convert_parquet_to_lerobot(parquet_dir: str, dataset: LeRobotDataset):
    parquet_files = sorted(Path(parquet_dir).glob("*.parquet"))
    print(f"[INFO] Found {len(parquet_files)} parquet episodes")

    for file_idx, parquet_path in enumerate(parquet_files):
        print(f"[INFO] Processing: {parquet_path}")
        table = pq.read_table(parquet_path)
        num_rows = table.num_rows

        for i in range(num_rows):
            row = table.slice(i, 1).to_pydict()
            
            img1_bytes = row["image_1"][0]
            img2_bytes = row["image_2"][0]
            img1 = cv2.imdecode(np.frombuffer(img1_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            img2 = cv2.imdecode(np.frombuffer(img2_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

            state = np.array(row["state"][0], dtype=np.float32)
            action = np.array(row["action"][0], dtype=np.float32)

            dataset.add_frame(
                {
                    "image_1": img1,
                    "image_2": img2,
                    "state": state,
                    "actions": action,
                }
            )

        prompt_bytes = row["prompt"][0]
        prompt_str = prompt_bytes.decode("utf-8")

        dataset.save_episode(task=prompt_str)
        print(f"[✔] Saved episode {file_idx:03d} with task prompt: {prompt_str}")

        # task_name = f"episode_{file_idx:03d}"
        # dataset.save_episode(task=task_name)
        # print(f"[✔] Saved episode: {task_name}")


def main(data_dir: str, *, push_to_hub: bool = False):
    output_path = LEROBOT_HOME / REPO_NAME
    print("output_path", output_path)
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="xarm6",
        fps=30,
        features={
            "image_1": {
                "dtype": "image",
                "shape": (1080, 1920, 3),
                "names": ["height", "width", "channel"],
            },
            "image_2": {
                "dtype": "image",
                "shape": (720, 1280, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    convert_parquet_to_lerobot(data_dir, dataset)
    dataset.consolidate(run_compute_stats=False)

    if push_to_hub:
        dataset.push_to_hub(
            tags=["parquet", "xarm", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
