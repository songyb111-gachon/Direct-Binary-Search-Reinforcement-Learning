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

import matplotlib.pyplot as plt

from env import BinaryHologramEnv

IPS = 256  #이미지 픽셀 사이즈
CH = 8  #채널
RW = 800  #보상

warnings.filterwarnings('ignore')

# 현재 날짜와 시간을 가져와 포맷 지정
current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

torch.backends.cudnn.enabled = False

class BinaryNet(nn.Module):
    def __init__(self, num_hologram, final='Sigmoid', in_planes=3,
                 channels=[32, 64, 128, 256, 512, 1024, 2048, 4096],
                 convReLU=True, convBN=True, poolReLU=True, poolBN=True,
                 deconvReLU=True, deconvBN=True):
        super(BinaryNet, self).__init__()

        def CRB2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, relu=True, bn=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            if relu:
                layers += [nn.Tanh()]
            if bn:
                layers += [nn.BatchNorm2d(num_features=out_channels)]

            cbr = nn.Sequential(*layers)  # *으로 list unpacking

            return cbr

        def TRB2d(in_channels, out_channels, kernel_size=2, stride=2, bias=True, relu=True, bn=True):
            layers = []
            layers += [nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                          kernel_size=2, stride=2, padding=0,
                                          bias=True)]
            if bn:
                layers += [nn.BatchNorm2d(num_features=out_channels)]
            if relu:
                layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)  # *으로 list unpacking

            return cbr

        self.enc1_1 = CRB2d(in_planes, channels[0], relu=convReLU, bn=convBN)
        self.enc1_2 = CRB2d(channels[0], channels[0], relu=convReLU, bn=convBN)
        self.pool1 = CRB2d(channels[0], channels[0], stride=2, relu=poolReLU, bn=poolBN)

        self.enc2_1 = CRB2d(channels[0], channels[1], relu=convReLU, bn=convBN)
        self.enc2_2 = CRB2d(channels[1], channels[1], relu=convReLU, bn=convBN)
        self.pool2 = CRB2d(channels[1], channels[1], stride=2, relu=poolReLU, bn=poolBN)

        self.enc3_1 = CRB2d(channels[1], channels[2], relu=convReLU, bn=convBN)
        self.enc3_2 = CRB2d(channels[2], channels[2], relu=convReLU, bn=convBN)
        self.pool3 = CRB2d(channels[2], channels[2], stride=2, relu=poolReLU, bn=poolBN)

        self.enc4_1 = CRB2d(channels[2], channels[3], relu=convReLU, bn=convBN)
        self.enc4_2 = CRB2d(channels[3], channels[3], relu=convReLU, bn=convBN)
        self.pool4 = CRB2d(channels[3], channels[3], stride=2, relu=poolReLU, bn=poolBN)

        self.enc5_1 = CRB2d(channels[3], channels[4], relu=convReLU, bn=convBN)
        self.enc5_2 = CRB2d(channels[4], channels[4], relu=convReLU, bn=convBN)

        self.deconv4 = TRB2d(channels[4], channels[3], relu=deconvReLU, bn=deconvBN, stride=2)
        self.dec4_1 = CRB2d(channels[4], channels[3], relu=convReLU, bn=convBN)
        self.dec4_2 = CRB2d(channels[3], channels[3], relu=convReLU, bn=convBN)

        self.deconv3 = TRB2d(channels[3], channels[2], relu=deconvReLU, bn=deconvBN, stride=2)
        self.dec3_1 = CRB2d(channels[3], channels[2], relu=convReLU, bn=convBN)
        self.dec3_2 = CRB2d(channels[2], channels[2], relu=convReLU, bn=convBN)

        self.deconv2 = TRB2d(channels[2], channels[1], relu=deconvReLU, bn=deconvBN, stride=2)
        self.dec2_1 = CRB2d(channels[2], channels[1], relu=convReLU, bn=convBN)
        self.dec2_2 = CRB2d(channels[1], channels[1], relu=convReLU, bn=convBN)

        self.deconv1 = TRB2d(channels[1], channels[0], relu=deconvReLU, bn=deconvBN, stride=2)
        self.dec1_1 = CRB2d(channels[1], channels[0], relu=convReLU, bn=convBN)
        self.dec1_2 = CRB2d(channels[0], channels[0], relu=convReLU, bn=convBN)

        self.classifier = CRB2d(channels[0], num_hologram, relu=False, bn=False)

    def forward(self, x):
        # Encoder
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)
        enc5_2 = self.enc5_2(enc5_1)

        deconv4 = self.deconv4(enc5_2)
        concat4 = torch.cat((deconv4, enc4_2), dim=1)
        dec4_1 = self.dec4_1(concat4)
        dec4_2 = self.dec4_2(dec4_1)

        deconv3 = self.deconv3(dec4_2)
        concat3 = torch.cat((deconv3, enc3_2), dim=1)
        dec3_1 = self.dec3_1(concat3)
        dec3_2 = self.dec3_2(dec3_1)

        deconv2 = self.deconv2(dec3_2)
        concat2 = torch.cat((deconv2, enc2_2), dim=1)
        dec2_1 = self.dec2_1(concat2)
        dec2_2 = self.dec2_2(dec2_1)

        deconv1 = self.deconv1(dec2_2)
        concat1 = torch.cat((deconv1, enc1_2), dim=1)
        dec1_1 = self.dec1_1(concat1)
        dec1_2 = self.dec1_2(dec1_1)

        # Final classifier
        out = self.classifier(dec1_2)
        out = nn.Sigmoid()(out)
        return out


model = BinaryNet(num_hologram=CH, in_planes=1, convReLU=False,
                  convBN=False, poolReLU=False, poolBN=False,
                  deconvReLU=False, deconvBN=False).cuda()
test = torch.randn(1, 1, IPS, IPS).cuda()
out = model(test)
print(out.shape)

class Dataset512(Dataset):
    def __init__(self, target_dir, meta, transform=None, isTrain=True, padding=0):
        self.target_dir = target_dir
        self.transform = transform
        self.meta = meta
        self.isTrain = isTrain
        self.target_list = sorted(glob.glob(target_dir+'*.png'))
        self.center_crop = torchvision.transforms.CenterCrop(IPS)
        self.random_crop = torchvision.transforms.RandomCrop((IPS, IPS))
        self.padding = padding

    def __len__(self):
        return len(self.target_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        target = tt.imread(self.target_list[idx], meta=self.meta, gray=True).unsqueeze(0)
        if target.shape[-1] < IPS or target.shape[-2] < IPS:
            target = torchvision.transforms.Resize(IPS)(target)
        if self.isTrain:
            target = self.random_crop(target)
            target = torchvision.transforms.functional.pad(target, (self.padding, self.padding, self.padding, self.padding))
        else:
            target = self.center_crop(target)
            target = torchvision.transforms.functional.pad(target, (self.padding, self.padding, self.padding, self.padding))
        # 데이터와 파일 경로를 함께 반환
        return target, self.target_list[idx]

# 에피소드 보상 로깅 콜백
class RewardLoggingCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.episode_rewards = []  # 각 에피소드 보상을 저장
        self.current_episode_reward = 0  # 현재 에피소드의 보상
        self.episode_count = 0  # 에피소드 수를 추적

    def _on_step(self) -> bool:
        # 현재 스텝의 보상을 누적
        reward = self.locals["rewards"]
        self.current_episode_reward += reward[0]  # 첫 번째 환경의 보상

        # 에피소드 종료 처리
        if self.locals["dones"][0]:  # 첫 번째 환경에서 에피소드 종료 시
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_count += 1

            if self.verbose > 0:
                print(f"\033[41mEpisode {self.episode_count}: Total Reward: {self.current_episode_reward:.2f}\033[0m")

            # 현재 에피소드 보상을 초기화
            self.current_episode_reward = 0

        return True  # 학습 계속

# 학습 종료 콜백
class StopOnEpisodeCallback(BaseCallback):
    def __init__(self, max_episodes, verbose=1):
        super(StopOnEpisodeCallback, self).__init__(verbose)
        self.max_episodes = max_episodes
        self.episode_count = 0  # 에피소드 수를 추적

    def _on_step(self) -> bool:
        # `dones`이 True일 때마다 에피소드 증가
        if self.locals.get("dones") is not None:
            self.episode_count += np.sum(self.locals["dones"])  # 에피소드 완료 횟수 추가

        if self.episode_count >= self.max_episodes:  # 최대 에피소드 도달 시 학습 종료
            print(f"Stopping training at episode {self.episode_count}")
            return False  # 학습 중단
        return True  # 학습 계속

batch_size = 1
target_dir = 'dataset6/'
#target_dir = '/nfs/dataset/DIV2K/DIV2K_train_HR/DIV2K_train_HR/'
valid_dir = '/nfs/dataset/DIV2K/DIV2K_valid_HR/DIV2K_valid_HR/'
meta = {'wl': (515e-9), 'dx': (7.56e-6, 7.56e-6)}  # 메타 정보
padding = 0

# Dataset512 클래스 사용
#train_dataset = Dataset512(target_dir=target_dir, meta=meta, isTrain=False, padding=padding) #센터크롭
train_dataset = Dataset512(target_dir=target_dir, meta=meta, isTrain=True, padding=padding) #랜덤크롭
valid_dataset = Dataset512(target_dir=valid_dir, meta=meta, isTrain=False, padding=padding)

# DataLoader 생성
#train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# BinaryNet 모델 로드
model = BinaryNet(num_hologram=CH, in_planes=1, convReLU=False, convBN=False,
                  poolReLU=False, poolBN=False, deconvReLU=False, deconvBN=False).cuda()
model.load_state_dict(torch.load('result_v/2024-12-19 20:37:52.499731_pre_reinforce_8_0.002/2024-12-19 20:37:52.499731_pre_reinforce_8_0.002'))
model.eval()

# 환경 생성에 새로운 데이터 로더 적용
env = BinaryHologramEnv(
    target_function=model,
    trainloader=train_loader,
    max_steps=10000,
    T_PSNR=30,
    T_steps=1,
    T_PSNR_DIFF=1/4,
    num_samples=10000
)

# PPO 모델 로드
ppo_model_path = "./env1_ppo_MlpPolicy_models/ppo_MlpPolicy_latest.zip"
ppo_model = PPO.load(ppo_model_path, env=env)

# 2. 단일 observation 얻기
obs, _ = env.reset()
obs_tensor = {k: torch.tensor(v, dtype=torch.float32).to(ppo_model.device).unsqueeze(0) for k, v in obs.items()}

# 3. 정책 분포 가져오기
with torch.no_grad():
    dist = ppo_model.policy.get_distribution(obs_tensor)
    probs = dist.distribution.probs.squeeze(0).cpu().numpy()  # shape: (action_dim,)

# 4. 확률 히스토그램 계산
bins = np.linspace(0, 1, 21)
hist, bin_edges = np.histogram(probs, bins=bins)

# 7. 가장 확률이 높은 행동 출력
top_idx = np.argmax(probs)
top_prob = probs[top_idx]

# index → (channel, row, col) 변환
channel = top_idx // (IPS * IPS)
pixel_index = top_idx % (IPS * IPS)
row = pixel_index // IPS
col = pixel_index % IPS

print("\n🔥 Most Probable Action:")
print(f"Index: {top_idx} | Channel: {channel}, Row: {row}, Col: {col} | Probability: {top_prob:.6f}")


# 5. 텍스트로 출력
print("\n🎯 Action Probability Histogram:")
for i in range(len(hist)):
    print(f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}): {hist[i]} actions")

# 6. 시각화
plt.figure(figsize=(10, 4))
plt.bar(range(len(hist)), hist, width=1.0, align='center')
plt.xticks(ticks=range(len(hist)), labels=[f"{bin_edges[i]:.2f}" for i in range(len(hist))], rotation=45)
plt.xlabel("Action Probability Bins")
plt.ylabel("Number of Actions")
plt.title("Histogram of PPO Policy Action Probabilities")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# 1. 최대 확률 구하기
max_prob = probs.max()

# 2. 0 ~ max_prob 사이를 100개 bin으로 나누기
bins = np.linspace(0, max_prob, 101)
hist, bin_edges = np.histogram(probs, bins=bins)

# 3. 텍스트 출력
print(f"\n🎯 Action Probability Histogram (0 ~ {max_prob:.6f}):")
for i in range(len(hist)):
    print(f"[{bin_edges[i]:.6f}, {bin_edges[i+1]:.6f}): {hist[i]} actions")

# 4. 시각화
plt.figure(figsize=(12, 4))
plt.bar(range(len(hist)), hist, width=1.0, align='center')
plt.xticks(ticks=np.linspace(0, 100, 11),  # 0,10,20,...100 구간 라벨만 출력
           labels=[f"{bin_edges[int(i)]:.4f}" for i in np.linspace(0, 100, 11)],
           rotation=45)
plt.xlabel("Action Probability Bins (0 ~ max_prob)")
plt.ylabel("Number of Actions")
plt.title("Histogram of PPO Action Probabilities (Split into 100 Bins)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()