import numpy as np

def dp_mdp_strategy(
        initial_price,  # 初始股价（元）
        total_shares,  # 总股数
        target_prices,  # 目标价列表（按从小到大排序）
        cumulative_probs,  # 各目标价的累积概率（至少达到该价格的概率）
        sell_shares_list,  # 各目标价的卖出股数（与目标价一一对应）
        annual_reinvest_rate,  # 卖出资金的年化再投资收益率（如15%则为0.15）
        time_interval=1  # 相邻阶段的时间间隔（月）
):
    """
    动态规划+马尔可夫决策模型计算策略一（分批卖出）和策略二（持有不动）的期望资产
    调整再投资收益计算方式：所有前序卖出资金统一按年化收益率计算到当前阶段的收益
    """
    n = len(target_prices)
    # 计算条件概率（从第i个目标价到第i+1个的概率）
    transition_probs = []
    for i in range(n - 1):
        trans_prob = cumulative_probs[i + 1] / cumulative_probs[i]
        transition_probs.append(trans_prob)
    transition_probs.append(1.0)  # 最后一个目标价无后续，概率设为1

    # 计算月化收益率（假设一年12个月）
    monthly_reinvest_rate = annual_reinvest_rate / 12

    # 逆向计算各阶段的期望资产
    remaining_shares = [0] * n
    cumulative_cash = [0.0] * n
    expected_assets_strategy1 = [0.0] * n  # 策略一（分批卖出）
    expected_assets_strategy2 = [0.0] * n  # 策略二（持有不动）

    # 最后一个阶段（i = n-1）
    i = n - 1
    remaining_shares[i] = total_shares - sum(sell_shares_list[:i + 1])
    # 累计卖出资金（所有前序卖出资金按年化收益率计算到当前阶段的收益）
    cash = 0.0
    for j in range(i + 1):
        # 从阶段j到阶段i的时间间隔（月）
        months = (i - j) * time_interval
        # 再投资收益 = 卖出资金 × (1 + 月化收益率)^月数
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

    # 初始阶段的期望资产
    initial_asset = initial_price * total_shares
    final_expected1 = (1 - cumulative_probs[0]) * initial_asset + cumulative_probs[0] * expected_assets_strategy1[0]
    final_expected2 = (1 - cumulative_probs[0]) * initial_asset + cumulative_probs[0] * expected_assets_strategy2[0]

    return {
        "target_prices": target_prices,
        "expected_strategy1": expected_assets_strategy1,
        "expected_strategy2": expected_assets_strategy2,
        "final_expected_strategy1": final_expected1,
        "final_expected_strategy2": final_expected2
    }


# --------------------------
# 参数设置（可自定义修改）
# --------------------------
initial_price = 45.69  # 初始股价
total_shares = 25500  # 总股数
target_prices = [46, 48, 50, 60, 70]  # 目标价列表
cumulative_probs = [0.95, 0.7, 0.5, 0.5, 0.5]  # 累积概率
sell_shares_list = [2550, 2550, 2550, 10000, 0]  # 各阶段卖出股数
annual_reinvest_rate = 0.15  # 年化再投资收益率（15%）
time_interval = 1  # 相邻阶段时间间隔（月）

# 运行模型
result = dp_mdp_strategy(
    initial_price=initial_price,
    total_shares=total_shares,
    target_prices=target_prices,
    cumulative_probs=cumulative_probs,
    sell_shares_list=sell_shares_list,
    annual_reinvest_rate=annual_reinvest_rate,
    time_interval=time_interval
)

# 输出结果
print("调整再投资收益后的模型结果：")
print(f"策略一（分批卖出）最终期望资产：{result['final_expected_strategy1']:.2f}元")
print(f"策略二（持有不动）最终期望资产：{result['final_expected_strategy2']:.2f}元")
print(f"策略一优势：{result['final_expected_strategy1'] - result['final_expected_strategy2']:.2f}元")

