from openpi_client import image_tools
from openpi_client import websocket_client_policy
import numpy as np
import time
import cv2
from xarm.wrapper import XArmAPI  
arm = XArmAPI('192.168.1.222')
arm.motion_enable(enable=True) 
arm.set_gripper_enable(enable=True)
arm.set_mode(6)  
arm.set_state(0)  





# Outside of episode loop, initialize the policy client.
# Point to the host and port of the policy server (localhost and 8000 are the defaults).
client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)


# 设置模拟参数
num_steps = 120  # 要推理多少步
resize_size = 224  # 图像 resize 到多大
action_horizon = 16  # 每次 infer 返回的动作步长，具体看模型设置（通常是16）
task_instruction = "pick the bottle and put it into the box"  # 给定固定指令文本

# OpenCV 视频捕获对象，分别用于两个摄像头
base_camera = cv2.VideoCapture("/dev/video0")  # 主摄像头
wrist_camera = cv2.VideoCapture("/dev/video6")  # 手腕摄像头

# 检查摄像头是否成功打开
if not base_camera.isOpened() or not wrist_camera.isOpened():
    print("错误：无法打开一个或两个摄像头。")
    exit()

init_qpos = np.array([14.1, -8, -24.7, 196.9, 62.3, -8.8])
# init_qpos = np.array([7, 19.7, -20.2, 182.1, 88.1, -1.7])
init_qpos = np.radians(init_qpos)
# print("init_qpos", init_qpos)
arm.set_servo_angle(angle=init_qpos,speed=8,is_radian=True)

for step in range(num_steps):
    # Inside the episode loop, construct the observation.
    # Resize images on the client side to minimize bandwidth / latency. Always return images in uint8 format.
    # We provide utilities for resizing images + uint8 conversion so you match the training routines.
    # The typical resize_size for pre-trained pi0 models is 224.
    # Note that the proprioceptive `state` can be passed unnormalized, normalization will be handled on the server side.

    # 生成随机图像（代替真实相机图像）
    # img = np.random.randint(0, 256, size=(1080, 1920, 3), dtype=np.uint8)         # 假装是 base camera
    # wrist_img = np.random.randint(0, 256, size=(720, 1280, 3), dtype=np.uint8)     # 假装是 wrist camera

    ret_base, img = base_camera.read()  # 从主摄像头读取
    ret_wrist, wrist_img = wrist_camera.read()  # 从手腕摄像头读取
    if not ret_base or not ret_wrist:
        print(f"错误：第 {step} 步时，未能从摄像头捕获图像。")
        break

     # 显示图像
    cv2.imshow('Base Camera (/dev/video0)', img)
    cv2.imshow('Wrist Camera (/dev/video6)', wrist_img)
    
    # 加一个短暂等待，允许窗口刷新
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("检测到 q 键，退出程序。")
        break
    
    _, curr_angles_rad = arm.get_servo_angle(is_radian=True)  # xarm6 有 7个自由度
    _, curr_gripper_state = arm.get_gripper_position()
    curr_state = np.append(np.rad2deg(curr_angles_rad[:-1]), curr_gripper_state)
    state = curr_state

    print(f"[Step {step}] Got state: {state}")

    
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
    # print("actions", action_chunk)

    
    # Execute the actions in the environment.
        # 只取前 20 个动作
    actions_to_execute = action_chunk[:20]  # (20, action_dim)

    # 依次执行这20个动作
    for i, action in enumerate(actions_to_execute):
        # 执行动作，你可以根据 action 的定义来做 set_servo_angle，或者 set_servo_cartesian等
        print(f"执行第 {i+1}/20 个动作: {action}")
        xarm_joint = action[:-1]  # 取出关节角度
        xarm_gripper = action[-1]
        print("xarm_joint",xarm_joint,"xarm_gripper", xarm_gripper)
        # 例子：如果 action 是关节角度增量，可以在这里控制
        # 比如 (伪代码)：

        
        arm.set_servo_angle(angle=xarm_joint,speed=8,is_radian=False)
        arm.set_gripper_position(pos=xarm_gripper, wait=False)

        print(f"step:{step},执行动作:{action}")
        time.sleep(1/30)  # 以 30Hz 频率执行
