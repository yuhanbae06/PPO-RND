from agents import *
from envs import *
from utils import *
from config import *
from torch.multiprocessing import Pipe

from tensorboardX import SummaryWriter

import cv2
import itertools
from typing import Iterable, Tuple, Union
import numpy as np
import pickle

from gym_minigrid.wrappers import RGBImgObsWrapper, RGBImgPartialObsWrapper, ImgObsWrapper

import inspect

def symbolic_partial_to_rgb(sym_img, tile_size=16, agent_dir=3):
    """
    sym_img: (7, 7, 3) uint8 from ImgObsWrapper (OBJECT_IDX, COLOR_IDX, STATE)
    returns: (7*tile_size, 7*tile_size, 3) uint8 RGB
    """
    # import Grid
    Grid = None
    try:
        from minigrid.core.grid import Grid            # Farama minigrid
    except Exception:
        from gym_minigrid.minigrid import Grid         # older gym_minigrid

    enc = sym_img.astype(np.uint8)

    # decode -> grid (handle both return styles)
    dec = Grid.decode(enc)
    grid = dec[0] if isinstance(dec, tuple) else dec

    # center of the 7x7 crop is the agent tile; partial obs is rotated so agent faces "right"
    H, W, _ = enc.shape
    cx, cy = W // 2, H-1

    # call render with whatever params your version expects
    try:
        return grid.render(tile_size, agent_pos=(cx, cy), agent_dir=agent_dir)
    except TypeError:
        # try positional args (older gym_minigrid): tile_size, agent_pos, agent_dir, highlight_mask
        try:
            return grid.render(tile_size, (cx, cy), agent_dir)
        except TypeError:
            try:
                highlight = np.zeros((W, H), dtype=bool)
                return grid.render(tile_size, (cx, cy), agent_dir, highlight)
            except TypeError:
                # final fallback: introspect signature & build kwargs
                sig = inspect.signature(grid.render)
                kwargs = {}
                for name in sig.parameters:
                    if name == "tile_size":        kwargs["tile_size"] = tile_size
                    elif name == "agent_pos":      kwargs["agent_pos"] = (cx, cy)
                    elif name == "agent_dir":      kwargs["agent_dir"] = agent_dir
                    elif name == "highlight_mask": kwargs["highlight_mask"] = np.zeros((W, H), dtype=bool)
                return grid.render(**kwargs)

def images_to_video(
    frames: Iterable[np.ndarray],
    out_path: str,
    fps: int = 30,
    codec: str = "mp4v",           # 'mp4v'가 가장 호환성 좋음. (H.264는 시스템에 따라 코덱 필요)
    channels_first: bool = False,  # (C,H,W) 입력이면 True
    from_rgb: bool = True,         # 프레임이 RGB라면 True (OpenCV는 BGR 기대)
    output_size: Union[None, Tuple[int, int]] = None,  # (width, height). None이면 첫 프레임 크기 사용
) -> None:
    """
    frames: 0~255 정수(uint8) ndarray들의 이터러블 (각 프레임)
            지원 형태: (H,W), (H,W,3), (H,W,4), 또는 (C,H,W) (channels_first=True)
    out_path: 저장할 비디오 경로 (예: 'out.mp4')
    fps: 초당 프레임 수
    codec: OpenCV FourCC 코드 ('mp4v', 'XVID', 'avc1' 등)
    output_size: (W,H). 지정하면 모든 프레임을 해당 크기로 리사이즈
    """
    it = iter(frames)
    try:
        first = next(it)
    except StopIteration:
        raise ValueError("frames 이 비어 있습니다.")

    def _prep(frame: np.ndarray) -> np.ndarray:
        # dtype 정규화
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        # 채널 순서 보정
        if channels_first and frame.ndim == 3:
            frame = np.transpose(frame, (1, 2, 0))  # (C,H,W) -> (H,W,C)

        # 그레이스케일/알파 처리
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)   # 1채널 -> 3채널
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]  # 알파 채널 제거

        # 색상 순서 (RGB -> BGR)
        if from_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        return frame

    first = _prep(first)
    if output_size is None:
        h, w = first.shape[:2]
        size = (w, h)
    else:
        size = (int(output_size[0]), int(output_size[1]))
        first = cv2.resize(first, size, interpolation=cv2.INTER_AREA)

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*codec),
        fps,
        size,  # (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("VideoWriter를 열 수 없습니다. 코덱/경로/권한을 확인하세요.")

    writer.write(first)

    for frame in itertools.chain([f for f in it]):  # 남은 프레임들
        frame = _prep(frame)
        if (frame.shape[1], frame.shape[0]) != size:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        writer.write(frame)

    writer.release()


def main():
    print({section: dict(config[section]) for section in config.sections()})
    env_id = default_config['EnvID']
    env_type = default_config['EnvType']

    if env_type == 'mario':
        env = BinarySpaceToDiscreteSpaceEnv(gym_super_mario_bros.make(env_id), COMPLEX_MOVEMENT)
    elif env_type == 'atari':
        env = gym.make(env_id)
    elif env_type == 'minigrid':
        env = gym.make(env_id)
        env = ImgObsWrapper(env)
    else:
        raise NotImplementedError
    input_size = env.observation_space.shape  # 4
    output_size = env.action_space.n  # 2

    if 'Breakout' in env_id:
        output_size -= 1

    env.close()

    is_render = True

    use_pred_cnn = default_config.getboolean('UsePredCNN')
    use_tar_cnn = default_config.getboolean('UseTarCNN')
    use_lora = default_config.getboolean('UseLoRA')
    use_pred_cnn_str = 'CNN' if use_pred_cnn else 'DNN'
    use_tar_cnn_str = 'CNN' if use_tar_cnn else 'DNN'

    model_path = 'models/{}.model'.format(env_id+use_pred_cnn_str+use_tar_cnn_str)
    predictor_path = 'models/{}.pred'.format(env_id+use_pred_cnn_str+use_tar_cnn_str)
    target_path = 'models/{}.target'.format(env_id+use_pred_cnn_str+use_tar_cnn_str)

    use_cuda = default_config.getboolean('UseGPU')
    use_gae = default_config.getboolean('UseGAE')
    use_noisy_net = default_config.getboolean('UseNoisyNet')

    lam = float(default_config['Lambda'])
    num_worker = 1

    num_step = int(default_config['NumStep'])

    ppo_eps = float(default_config['PPOEps'])
    epoch = int(default_config['Epoch'])
    mini_batch = int(default_config['MiniBatch'])
    batch_size = int(num_step * num_worker / mini_batch)
    learning_rate = float(default_config['LearningRate'])
    entropy_coef = float(default_config['Entropy'])
    gamma = float(default_config['Gamma'])
    int_gamma = float(default_config['IntGamma'])
    clip_grad_norm = float(default_config['ClipGradNorm'])
    ext_coef = float(default_config['ExtCoef'])
    int_coef = float(default_config['IntCoef'])

    sticky_action = False
    action_prob = float(default_config['ActionProb'])
    life_done = default_config.getboolean('LifeDone')

    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(shape=(1, 1, 7, 7, 3))
    pre_obs_norm_step = int(default_config['ObsNormStep'])
    discounted_reward = RewardForwardFilter(int_gamma)

    max_update = int(default_config['MaxUpdate'])
    R_LORA = int(default_config['RLoRA'])
    ENV_SEED = int(default_config['EnvSeed'])
    ALPHA = float(default_config['Alpha'])
    MODEL_WIDTH = int(default_config['ModelWidth'])

    agent = RNDAgent

    if default_config['EnvType'] == 'atari':
        env_type = AtariEnvironment
    elif default_config['EnvType'] == 'mario':
        env_type = MarioEnvironment
    elif default_config['EnvType'] == 'minigrid':
        env_type = MiniGridEnvironment
    else:
        raise NotImplementedError

    agent = agent(
        input_size,
        output_size,
        num_worker,
        num_step,
        gamma,
        lam=lam,
        learning_rate=learning_rate,
        ent_coef=entropy_coef,
        clip_grad_norm=clip_grad_norm,
        epoch=epoch,
        batch_size=batch_size,
        ppo_eps=ppo_eps,
        use_cuda=use_cuda,
        use_gae=use_gae,
        use_noisy_net=use_noisy_net,
        use_pred_cnn=use_pred_cnn,
        use_tar_cnn=use_tar_cnn,
        use_lora=use_lora,
        r_lora = R_LORA,
        alpha = ALPHA,
        model_width = MODEL_WIDTH
    )

    print('Loading Pre-trained model....')
    if use_cuda:
        agent.model.load_state_dict(torch.load(model_path))
        agent.rnd.predictor.load_state_dict(torch.load(predictor_path))
        agent.rnd.target.load_state_dict(torch.load(target_path))
    else:
        agent.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        agent.rnd.predictor.load_state_dict(torch.load(predictor_path, map_location='cpu'))
        agent.rnd.target.load_state_dict(torch.load(target_path, map_location='cpu'))
    print('End load...')

    works = []
    parent_conns = []
    child_conns = []
    for idx in range(num_worker):
        parent_conn, child_conn = Pipe()
        work = env_type(env_id, is_render, idx, child_conn, sticky_action=sticky_action, p=action_prob,
                        life_done=life_done, env_seed=ENV_SEED)
        work.start()
        works.append(work)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    states = np.zeros([num_worker, 4, 7, 7, 3])

    steps = 0
    rall = 0
    rd = False
    intrinsic_reward_list = []
    frames = []
    while not rd:
        steps += 1
        actions, value_ext, value_int, policy = agent.get_action(np.float32(states) / 10.)

        for parent_conn, action in zip(parent_conns, actions):
            parent_conn.send(action)

        next_states, rewards, dones, real_dones, log_rewards, next_obs = [], [], [], [], [], []
        for parent_conn in parent_conns:
            s, r, d, rd, lr = parent_conn.recv()
            rall += r
            next_states = s.reshape([1, 4, 7, 7, 3])
            next_obs = s[3, :, :].reshape([1, 1, 7, 7, 3])

        # total reward = int reward + ext Reward
        img = symbolic_partial_to_rgb(next_obs[0, 0, :, :, :], tile_size=32)
        frames.append(img)
        intrinsic_reward = agent.compute_intrinsic_reward(next_obs)
        intrinsic_reward_list.append(intrinsic_reward)
        states = next_states[:, :, :, :]

        if rd:
            images_to_video(frames[:-1], "out_" + str(ENV_SEED) + ".mp4", fps=4)
            intrinsic_reward_list = (intrinsic_reward_list - np.mean(intrinsic_reward_list)) / np.std(
                intrinsic_reward_list)
            with open('int_reward', 'wb') as f:
                pickle.dump(intrinsic_reward_list, f)
            steps = 0
            rall = 0


if __name__ == '__main__':
    main()
