from openpi_client import image_tools
from openpi_client import websocket_client_policy
import numpy as np
import time
# Outside of episode loop, initialize the policy client.
# Point to the host and port of the policy server (localhost and 8000 are the defaults).
client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)

# 设置模拟参数
num_steps = 10  # 要推理多少步
resize_size = 224  # 图像 resize 到多大
action_horizon = 16  # 每次 infer 返回的动作步长，具体看模型设置（通常是16）
task_instruction = "pick the bottle and put it into the box"  # 给定固定指令文本

for step in range(num_steps):
    # Inside the episode loop, construct the observation.
    # Resize images on the client side to minimize bandwidth / latency. Always return images in uint8 format.
    # We provide utilities for resizing images + uint8 conversion so you match the training routines.
    # The typical resize_size for pre-trained pi0 models is 224.
    # Note that the proprioceptive `state` can be passed unnormalized, normalization will be handled on the server side.

    # 生成随机图像（代替真实相机图像）
    img = np.random.randint(0, 256, size=(1080, 1920, 3), dtype=np.uint8)         # 假装是 base camera
    wrist_img = np.random.randint(0, 256, size=(720, 1280, 3), dtype=np.uint8)     # 假装是 wrist camera

    # 随机生成关节状态（代替真实机械臂状态）
    state = np.random.uniform(low=-1.0, high=1.0, size=(7,))  # xarm6 有 7个自由度


    
    observation = {
        "observation/image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        ),
        "observation/wrist_image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, 224, 224)
        ),
        "observation/state": state,
        "prompt": task_instruction,
    }

    # Call the policy server with the current observation.
    # This returns an action chunk of shape (action_horizon, action_dim).
    # Note that you typically only need to call the policy every N steps and execute steps
    # from the predicted action chunk open-loop in the remaining steps.
    action_chunk = client.infer(observation)["actions"]
    print(f"[Step {step}] Got action chunk of shape {action_chunk.shape}")
    print("actions", action_chunk)
    # Execute the actions in the environment.
    ...
