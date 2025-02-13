import numpy as np
import matplotlib.pyplot as plt

# 기준점 (예시: 목표 달성률과 해당 보상 값)
# 목표 달성률이 높을수록 보상이 낮고, 낮을수록 보상이 높은 형태로 설정합니다.
# 예를 들어, 10000일 때 -0.5, 9000일 때 -0.48, 8000일 때 -0.45, 5000일 때 -0.35,
# 2500일 때 0, 1일 때 1의 보상을 부여합니다.
success_ratios = np.array([10000, 9000, 8000, 5000, 2500, 1])
rewards = np.array([-0.5, -0.48, -0.45, -0.35, 0, 1])

# 5차 다항식으로 보상 함수 추정
# (6개의 데이터 포인트에 대해 5차 다항식은 정확히 통과하는 유일한 함수입니다.)
degree = len(success_ratios) - 1  # degree = 5
coefficients = np.polyfit(success_ratios, rewards, degree)
reward_poly = np.poly1d(coefficients)

# 다항식 형태의 보상 함수 식을 문자열로 생성 (터미널 및 디버깅용)
terms = []
for i, coeff in enumerate(coefficients):
    power = degree - i
    if power > 1:
        terms.append(f"{coeff:.2e}x^{power}")
    elif power == 1:
        terms.append(f"{coeff:.2e}x")
    else:
        terms.append(f"{coeff:.2e}")
equation_text = "y = " + " + ".join(terms)
print("Reward Function Equation:")
print(equation_text)


def reward_function(success_ratio):
    """
    강화학습에서 사용할 보상 함수입니다.

    Parameters:
        success_ratio (float or np.array): 목표 달성률 (예: 1, 2500, 10000 등)

    Returns:
        float or np.array: 입력 success_ratio에 대응하는 보상 값
    """
    return reward_poly(success_ratio)


# 예시: 성공률 5000에 대한 보상값 계산
example_ratio = 5000
print(f"\nReward for success ratio {example_ratio}: {reward_function(example_ratio):.4f}")

# -- 아래 코드는 선택 사항입니다. 보상 함수를 시각화하여 형태를 확인할 수 있습니다. --
# RL 알고리즘에서 보상 함수로 바로 사용하려면 reward_function()을 호출하면 됩니다.

# 보상 함수의 형태를 확인하기 위한 그래프 (x축: success_ratio, y축: reward)
x_vals = np.linspace(1, 10000, 1000)
reward_vals = reward_function(x_vals)

plt.figure(figsize=(8, 6))
plt.plot(x_vals, reward_vals, label="5th Degree Reward Function", color='orange')
plt.scatter(success_ratios, rewards, color='red', label="Target Points")
plt.xlabel("Success Ratio")
plt.ylabel("Reward")
plt.title("Reinforcement Learning Reward Function")
plt.legend()
plt.grid()
plt.show()