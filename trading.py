import numpy as np


def dp_mdp_strategy_with_downside(
        initial_price,  # 初始股价（元）
        total_shares,  # 总股数
        target_prices,  # 目标价列表（按从小到大排序，包含下跌情况）
        cumulative_probs,  # 各目标价的累积概率（至少达到该价格的概率）
        sell_shares_list,  # 各目标价的卖出股数（与目标价一一对应）
        annual_reinvest_rate,  # 卖出资金的年化再投资收益率（如15%则为0.15）
        time_interval=1  # 相邻阶段的时间间隔（月）
):
    """
    动态规划+马尔可夫决策模型，考虑股价下跌情况，计算不同最终状态的期望收益
    """
    n = len(target_prices)
    # 计算条件概率（从第i个目标价到第i+1个的概率）
    transition_probs = []
    for i in range(n - 1):
        trans_prob = cumulative_probs[i + 1] / cumulative_probs[i]
        transition_probs.append(trans_prob)
    transition_probs.append(1.0)  # 最后一个目标价无后续，概率设为1

    # 计算月化收益率
    monthly_reinvest_rate = annual_reinvest_rate / 12

    # 逆向计算各阶段的期望资产
    remaining_shares = [0] * n
    cumulative_cash = [0.0] * n
    expected_assets_strategy1 = [0.0] * n  # 策略一（分批卖出）
    expected_assets_strategy2 = [0.0] * n  # 策略二（持有不动）

    # 最后一个阶段（i = n-1）
    i = n - 1
    remaining_shares[i] = total_shares - sum(sell_shares_list[:i + 1])
    cash = 0.0
    for j in range(i + 1):
        months = (i - j) * time_interval
        cash += sell_shares_list[j] * target_prices[j] * (1 + monthly_reinvest_rate) ** months
    cumulative_cash[i] = cash
    expected_assets_strategy1[i] = remaining_shares[i] * target_prices[i] + cumulative_cash[i]
    expected_assets_strategy2[i] = total_shares * target_prices[i]

    # 逆向计算前n-1个阶段
    for i in range(n - 2, -1, -1):
        remaining_shares[i] = total_shares - sum(sell_shares_list[:i + 1])
        cash = 0.0
        for j in range(i + 1):
            months = (i - j) * time_interval
            cash += sell_shares_list[j] * target_prices[j] * (1 + monthly_reinvest_rate) ** months
        cumulative_cash[i] = cash

        # 策略一的期望资产
        expected_assets_strategy1[i] = (
                (1 - transition_probs[i]) * (remaining_shares[i] * target_prices[i] + cumulative_cash[i])
                + transition_probs[i] * expected_assets_strategy1[i + 1]
        )
        # 策略二的期望资产
        expected_assets_strategy2[i] = (
                (1 - transition_probs[i]) * (total_shares * target_prices[i])
                + transition_probs[i] * expected_assets_strategy2[i + 1]
        )

    # 计算各最终状态的实际收益（非期望收益）
    final_assets_strategy1 = []
    final_assets_strategy2 = []
    for i in range(n):
        # 若最终停在状态i（未继续上涨）
        remaining = total_shares - sum(sell_shares_list[:i + 1])
        cash = 0.0
        for j in range(i + 1):
            months = (i - j) * time_interval
            cash += sell_shares_list[j] * target_prices[j] * (1 + monthly_reinvest_rate) ** months
        final_assets_strategy1.append(remaining * target_prices[i] + cash)
        final_assets_strategy2.append(total_shares * target_prices[i])

    # 初始阶段的期望资产
    initial_asset = initial_price * total_shares
    final_expected1 = (1 - cumulative_probs[0]) * initial_asset + cumulative_probs[0] * expected_assets_strategy1[0]
    final_expected2 = (1 - cumulative_probs[0]) * initial_asset + cumulative_probs[0] * expected_assets_strategy2[0]

    return {
        "target_prices": target_prices,
        "final_assets_strategy1": final_assets_strategy1,  # 各最终状态的策略一资产
        "final_assets_strategy2": final_assets_strategy2,  # 各最终状态的策略二资产
        "expected_strategy1": expected_assets_strategy1,  # 各阶段的期望资产（策略一）
        "expected_strategy2": expected_assets_strategy2,  # 各阶段的期望资产（策略二）
        "final_expected_strategy1": final_expected1,  # 策略一总期望资产
        "final_expected_strategy2": final_expected2  # 策略二总期望资产
    }


# --------------------------
# 下跌场景参数设置
# --------------------------
initial_price = 10.0  # 初始股价
total_shares = 20000  # 总股数
target_prices = [6, 8, 10, 12, 14]  # 目标价列表（包含下跌情况）
cumulative_probs = [0.8, 0.9, 0.9, 0.8, 0.6]  # 累积概率（至少达到该价格的概率）
sell_shares_list = [3000, 3000, 2000, 2000, 2000]  # 各阶段卖出股数（下跌时多卖，上涨时少卖）
annual_reinvest_rate = 0.15  # 年化再投资收益率（15%）
time_interval = 1  # 相邻阶段时间间隔（月）

# 运行模型
result = dp_mdp_strategy_with_downside(
    initial_price=initial_price,
    total_shares=total_shares,
    target_prices=target_prices,
    cumulative_probs=cumulative_probs,
    sell_shares_list=sell_shares_list,
    annual_reinvest_rate=annual_reinvest_rate,
    time_interval=time_interval
)

# 输出各最终状态的收益
print("各最终状态的实际收益：")
for i, price in enumerate(target_prices):
    print(f"最终价格 {price} 元：")
    print(f"  策略一资产：{result['final_assets_strategy1'][i]:.2f} 元")
    print(f"  策略二资产：{result['final_assets_strategy2'][i]:.2f} 元")
    print(f"  策略一优势：{result['final_assets_strategy1'][i] - result['final_assets_strategy2'][i]:.2f} 元")

# 输出总期望收益
print("\n总期望收益：")
print(f"策略一总期望资产：{result['final_expected_strategy1']:.2f} 元")
print(f"策略二总期望资产：{result['final_expected_strategy2']:.2f} 元")
print(f"策略一总期望优势：{result['final_expected_strategy1'] - result['final_expected_strategy2']:.2f} 元")