import dataclasses
import enum
import logging
import time

import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    XARM6 = "xarm6"

@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    
    env: EnvMode = EnvMode.XARM6
    num_steps: int = 100


def main(args: Args) -> None:
    obs_fn = {
        EnvMode.ALOHA: _random_observation_aloha,
        EnvMode.ALOHA_SIM: _random_observation_aloha,
        EnvMode.DROID: _random_observation_droid,
        EnvMode.LIBERO: _random_observation_libero,
        EnvMode.XARM6: _random_observation_xarm6,
    }[args.env]

    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logging.info(f"Server metadata: {policy.get_server_metadata()}")

    # Send 1 observation to make sure the model is loaded.
    policy.infer(obs_fn())

    start = time.time()
    for _ in range(args.num_steps):
        actions = policy.infer(obs_fn())
        print(actions)
    end = time.time()

    print(f"Total time taken: {end - start:.2f} s")
    print(f"Average inference time: {1000 * (end - start) / args.num_steps:.2f} ms")


def _random_observation_aloha() -> dict:
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


def _random_observation_droid() -> dict:
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }


def _random_observation_libero() -> dict:
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }

def _random_observation_xarm6() -> dict:
    """新增：随机生成符合 xarm6 输入格式的 observation"""
    return {
        "observation/state": np.random.rand(7),  # xarm6 是7维 state
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),  # base camera
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),  # wrist camera
        "prompt": "pick the bottle and put in box",  # 也可以写成随机的
    }
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
