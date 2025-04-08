import os
import glob
import shutil
from datetime import datetime
import time
import warnings

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import Dataset, DataLoader

import torchvision

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

import torchOptics.optics as tt
import torchOptics.metrics as tm

IPS = 256  #이미지 픽셀 사이즈
CH = 8  #채널
RW = 800  #보상

warnings.filterwarnings('ignore')

# 현재 날짜와 시간을 가져와 포맷 지정
current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# BinaryHologramEnv 클래스
class BinaryHologramEnv(gym.Env):
    def __init__(self, target_function, trainloader, max_steps=10000, T_PSNR=30, T_steps=1, T_PSNR_DIFF=1/4, num_samples=10000):
        super(BinaryHologramEnv, self).__init__()

        # 관찰 공간 정보
        self.observation_space = spaces.Dict({
            "state_record": spaces.Box(low=0, high=1, shape=(1, CH, IPS, IPS), dtype=np.int8),
            "state": spaces.Box(low=0, high=1, shape=(1, CH, IPS, IPS), dtype=np.int8),
            "pre_model": spaces.Box(low=0, high=1, shape=(1, CH, IPS, IPS), dtype=np.float32),
            "recon_image": spaces.Box(low=0, high=1, shape=(1, 1, IPS, IPS), dtype=np.float32),
            "target_image": spaces.Box(low=0, high=1, shape=(1, 1, IPS, IPS), dtype=np.float32),
        })

        # 행동 공간: 픽셀 하나를 선택하는 인덱스 (CH * IPS * IPS)
        self.num_pixels = CH * IPS * IPS
        self.action_space = spaces.Discrete(self.num_pixels)

        # 타겟 함수와 데이터 로더 설정
        self.target_function = target_function
        self.trainloader = trainloader

        # 환경 설정
        self.max_steps = max_steps
        self.T_PSNR = T_PSNR
        self.T_steps = T_steps
        self.T_PSNR_DIFF_o = T_PSNR_DIFF
        self.T_PSNR_DIFF = None
        self.num_samples = num_samples
        self.target_step = self.T_PSNR_DIFF_o * self.num_samples

        # 학습 상태 초기화
        self.state = None
        self.state_record = None
        self.observation = None
        self.steps = None
        self.psnr_sustained_steps = None
        self.flip_count = None
        self.start_time = None
        self.next_print_thresholds = 0
        self.total_start_time = None
        self.target_image_np = None
        self.initial_psnr = None

        # 최고 PSNR_DIFF 추적 변수
        self.max_psnr_diff = float('-inf')  # 가장 높은 PSNR_DIFF를 추적

        # PSNR 저장 변수
        self.previous_psnr = None

        # 데이터 로더에서 첫 배치 설정
        self.data_iter = iter(self.trainloader)
        self.target_image = None

        # 에피소드 카운트
        self.episode_num_count = 0

    def _calculate_pixel_importance(self, binary, z):
        #랜덤으로 1만 개 픽셀 플립 후 PSNR 변화량을 순위화 및 양수 변화량 총합 계산
        num_samples = self.num_samples
        psnr_changes = []
        positive_psnr_sum = 0  # 양수 PSNR 변화량 총합 초기화

        for _ in range(num_samples):
            random_action = np.random.randint(self.num_pixels)
            channel, pixel_index = divmod(random_action, IPS * IPS)
            row, col = divmod(pixel_index, IPS)

            # 픽셀 플립
            self.state[0, channel, row, col] = 1 - self.state[0, channel, row, col]

            binary_temp = torch.tensor(self.state, dtype=torch.float32).cuda()
            binary_temp = tt.Tensor(binary_temp, meta={'dx': (7.56e-6, 7.56e-6), 'wl': 515e-9})

            sim_temp = tt.simulate(binary_temp, z).abs() ** 2
            result_temp = torch.mean(sim_temp, dim=1, keepdim=True)
            psnr_temp = tt.relativeLoss(result_temp, self.target_image, tm.get_PSNR)

            psnr_change = psnr_temp - self.initial_psnr
            psnr_changes.append(psnr_change)

            # 양수 PSNR 변화량만 누적
            if psnr_change > 0:
                positive_psnr_sum += psnr_change

            # 상태 롤백
            self.state[0, channel, row, col] = 1 - self.state[0, channel, row, col]

        # PSNR 변화량을 기준으로 순위 매기기 (오름차순 정렬)
        sorted_indices = np.argsort(psnr_changes)
        importance_ranks = np.zeros(num_samples)

        for rank, idx in enumerate(sorted_indices):
            # 순위(rank)를 [10000, 1] 범위의 x값으로 선형 변환
            # rank = 0  -> x_val = 10000
            # rank = num_samples-1 -> x_val = 1
            x_val = rank + 1
            # 다항식 보상 함수를 사용하여 보상값 계산
            importance_ranks[idx] = 1/x_val

        return psnr_changes, importance_ranks, positive_psnr_sum,

    def reset(self, seed=None, options=None, z=2e-3):
        torch.cuda.empty_cache()

        self.episode_num_count += 1  # Increment episode count at the start of each reset

        # 이터레이터에서 다음 데이터를 가져옴
        try:
            self.target_image, self.current_file = next(self.data_iter)
        except StopIteration:
            # 데이터셋 끝에 도달하면 이터레이터를 다시 생성하고 처음부터 다시 시작
            print(f"\033[40;93m[INFO] Reached the end of dataset. Restarting from the beginning.\033[0m")
            self.data_iter = iter(self.trainloader)
            self.target_image, self.current_file = next(self.data_iter)

        print(f"\033[40;93m[Episode Start] Currently using dataset file: {self.current_file}, Episode count: {self.episode_num_count}\033[0m")

        self.target_image = self.target_image.cuda()
        self.target_image_np = self.target_image.cpu().numpy()

        with torch.no_grad():
            model_output = self.target_function(self.target_image)
        self.observation = model_output.cpu().numpy()  # (1, CH, IPS, IPS)

        # 매 에피소드마다 초기화
        self.max_psnr_diff = float('-inf')
        self.steps = 0
        self.flip_count = 0
        self.psnr_sustained_steps = 0
        self.next_print_thresholds = 0

        self.state = (self.observation >= 0.5).astype(np.int8)  # 초기 Binary state
        self.state_record = np.zeros_like(self.state)  # 플립 횟수를 저장하기 위한 배열 초기화

        binary = torch.tensor(self.state, dtype=torch.float32).cuda()  # (1, CH, IPS, IPS)
        binary = tt.Tensor(binary, meta={'dx': (7.56e-6, 7.56e-6), 'wl': 515e-9})  # meta 정보 포함

        # 시뮬레이션
        sim = tt.simulate(binary, z).abs()**2
        result = torch.mean(sim, dim=1, keepdim=True)

        # MSE 및 PSNR 계산
        mse = tt.relativeLoss(result, self.target_image, F.mse_loss).detach().cpu().numpy()
        self.initial_psnr = tt.relativeLoss(result, self.target_image, tm.get_PSNR)  # 초기 PSNR 저장
        self.previous_psnr = self.initial_psnr # 초기 PSNR 저장

        # 1만 개 픽셀 플립 후 PSNR 변화량 순위화 및 양수 변화량 총합 계산
        rw_start_time = time.time()
        self.psnr_change_list, self.importance_ranks, positive_psnr_sum = self._calculate_pixel_importance(binary, z)
        data_processing_time = time.time() - rw_start_time
        print(
            f"\nTime taken for psnr_change_list: {data_processing_time:.2f} seconds"
        )

        self.T_PSNR_DIFF = self.T_PSNR_DIFF_o * positive_psnr_sum
        print(f"\033[94m[Dynamic Threshold] T_PSNR_DIFF set to: {self.T_PSNR_DIFF:.6f}\033[0m")

        obs = {"state_record": self.state_record,
               "state": self.state,
               "pre_model": self.observation,
               "recon_image": result.cpu().numpy(),
               "target_image": self.target_image_np,
               }

        print(
            f"\033[92mInitial PSNR: {self.initial_psnr:.6f}\033[0m"
            f"\nInitial MSE: {mse:.6f}\033[0m"
        )

        # 다음 출력 기준 PSNR 값 리스트 설정 (0.01 단위로 증가)
        self.next_print_thresholds = [self.initial_psnr + i * 0.01 for i in range(1, 21)]  # 최대 0.1 상승까지 출력

        self.total_start_time = time.time()

        return obs, {"state": self.state}

    def step(self, action, z=2e-3):
        self.steps += 1

        # 행동을 기반으로 픽셀 좌표 계산
        channel = action // (IPS * IPS)
        pixel_index = action % (IPS * IPS)
        row = pixel_index // IPS
        col = pixel_index % IPS

        # 상태 변경
        self.state[0, channel, row, col] = 1 - self.state[0, channel, row, col]
        self.state_record[0, channel, row, col] = self.state_record[0, channel, row, col] + 1

        self.flip_count += 1  # 플립 증가

        # 시뮬레이션 수행
        binary_after = torch.tensor(self.state, dtype=torch.float32).cuda()
        binary_after = tt.Tensor(binary_after, meta={'dx': (7.56e-6, 7.56e-6), 'wl': 515e-9})
        sim_after = tt.simulate(binary_after, z).abs()**2
        result_after = torch.mean(sim_after, dim=1, keepdim=True)
        psnr_after = tt.relativeLoss(result_after, self.target_image, tm.get_PSNR)

        obs = {"state_record": self.state_record,
               "state": self.state,
               "pre_model": self.observation,
               "recon_image": result_after.cpu().numpy(),
               "target_image": self.target_image_np,
               }

        # PSNR 변화량 계산
        psnr_change = psnr_after - self.previous_psnr
        psnr_diff = psnr_after - self.initial_psnr

        # 데이터 내에서 PSNR 변화량이 0에 가장 가까운 샘플(기준 샘플)을 찾음
        neutral_index = np.argmin(np.abs(np.array(self.psnr_change_list)))
        neutral_psnr_change = self.psnr_change_list[neutral_index]
        neutral_rank = self.importance_ranks[neutral_index]

        # 기존 _calculate_pixel_importance에서 저장한 값을 사용합니다.
        # (여기서는 절대값 기준으로 0에 가까운 샘플을 기준으로 보상 조건을 결정)
        # reward_probability (즉, 보상을 받을 확률)
        reward_probability = (neutral_index + 1) / self.num_pixels
        # 예를 들어 self.neutral_index가 1500이면 reward_probability ≈ 0.15 (15%)
        # 최종 보상 상수는 미리 reset 시에 self.final_reward_constant로 정의 (예: 1.0)
        scaling_factor = 1.0 / reward_probability
        # scaling_factor = final_reward_constant * num_samples / (neutral_index + 1)

        # 현재 액션에 대한 보상 계산:
        # "0에 더 가까운" (즉, np.abs(psnr_change)가 작으면 보상이 주어짐)
        # (원래 코드는 psnr_change가 음수면 롤백하는 식이었으므로, 조건은 필요에 따라 조정하세요.)
        if np.abs(psnr_change) < np.abs(neutral_psnr_change):
            # 현재 액션의 psnr_change와 가장 유사한(순위가 매겨진) 값의 인덱스를 찾음
            closest_index = np.argmin(np.abs(np.array(self.psnr_change_list) - psnr_change))
            # 기존 순위 기반 보상에 scaling_factor를 곱하여 최종 보상 산출
            reward = self.importance_ranks[closest_index] * scaling_factor
        else:
            reward = 0.0

        # psnr_change가 음수인 경우 상태 롤백 수행
        if psnr_change < 0:

            self.state[0, channel, row, col] = 1 - self.state[0, channel, row, col]
            self.flip_count -= 1

            return obs, reward, False, False, {}

        self.max_psnr_diff = max(self.max_psnr_diff, psnr_diff)  # 최고 PSNR_DIFF 업데이트

        success_ratio = self.flip_count / self.steps if self.steps > 0 else 0

        # 출력 추가 (0.01 PSNR 상승마다 출력)
        while self.next_print_thresholds and psnr_after >= self.next_print_thresholds[0]:
            self.next_print_thresholds.pop(0)
            data_processing_time = time.time() - self.total_start_time
            print(
                f"Step: {self.steps:<6} | Initial PSNR: {self.initial_psnr:.6f}"
                f"\nPSNR After: {psnr_after:.6f} | Change: {psnr_change:.6f} | Diff: {psnr_diff:.6f}"
                f"\nReward: {reward:.2f} | Success Ratio: {success_ratio:.6f} | Flip Count: {self.flip_count}"
                f"\nFlip Pixel: Channel={channel}, Row={row}, Col={col}"
                f"\nTime taken for this data: {data_processing_time:.2f} seconds"
            )

        self.previous_psnr = psnr_after

        if psnr_diff >= self.T_PSNR_DIFF or (psnr_after >= self.T_PSNR and psnr_diff < 0.1):
            data_processing_time = time.time() - self.total_start_time
            print(
                f"Step: {self.steps:<6} | Initial PSNR: {self.initial_psnr:.6f}"
                f"\nPSNR After: {psnr_after:.6f} | Change: {psnr_change:.6f} | Diff: {psnr_diff:.6f}"
                f"\nReward: {reward:.2f} | Success Ratio: {success_ratio:.6f} | Flip Count: {self.flip_count}"
                f"\nFlip Pixel: Channel={channel}, Row={row}, Col={col}"
                f"\nTime taken for this data: {data_processing_time:.2f} seconds"
            )
            self.psnr_sustained_steps += 1

            if self.psnr_sustained_steps >= self.T_steps and psnr_diff >= self.T_PSNR_DIFF:   # 성공 에피소드 조건
                # 스텝에 따른 추가 보상 계산 (선형 보상)
                m = -1000 / (3 * self.target_step)
                additional_reward = 100 + m * (self.steps - (2 / 5) * self.target_step)
                reward += additional_reward

        if self.steps >= self.max_steps:
            # 현재 PSNR 값이 출력 기준을 충족했는지 확인
            data_processing_time = time.time() - self.total_start_time
            print(
                f"Step: {self.steps:<6} | Initial PSNR: {self.initial_psnr:.6f}"
                f"\nPSNR After: {psnr_after:.6f} | Change: {psnr_change:.6f} | Diff: {psnr_diff:.6f}"
                f"\nReward: {reward:.2f} | Success Ratio: {success_ratio:.6f} | Flip Count: {self.flip_count}"
                f"\nFlip Pixel: Channel={channel}, Row={row}, Col={col}"
                f"\nTime taken for this data: {data_processing_time:.2f} seconds"
            )
            # 스텝에 따른 추가 보상 계산 (선형 보상)
            m = -1000 / (3 * self.target_step)
            additional_reward = 100 + m * (self.steps - (2 / 5) * self.target_step)
            reward += additional_reward

        # 성공 종료 조건: PSNR >= T_PSNR 또는 PSNR_DIFF >= T_PSNR_DIFF
        terminated = self.psnr_sustained_steps >= self.T_steps and psnr_diff >= self.T_PSNR_DIFF
        truncated = self.steps >= self.max_steps

        return obs, reward, terminated, truncated, {}  # 빈 딕셔너리 반환