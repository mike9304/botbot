"""Reinforcement Learning components for trading agent evolution.

Implements a lightweight RL framework that can work alongside the genetic algorithm.
The RL agent learns position sizing, entry timing, and exit timing.

References:
- "Deep Reinforcement Learning for Automated Stock Trading" - Yang et al. (2020)
- FinRL: A Deep RL Library for Automated Stock Trading (Liu et al., 2020)
- "Practical Deep RL Approach for Stock Trading" (2019)
- PPO: "Proximal Policy Optimization Algorithms" - Schulman et al. (2017)
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RLState:
    """State representation for RL agent.

    Features:
    - Price returns (multiple timeframes)
    - Volatility
    - RSI
    - Volume ratio
    - Current position info
    - Account metrics
    """

    price_returns: list[float] = field(default_factory=list)  # Last N returns
    volatility: float = 0.0
    rsi: float = 50.0
    volume_ratio: float = 1.0
    has_position: bool = False
    position_pnl_pct: float = 0.0
    account_equity_pct: float = 1.0
    drawdown: float = 0.0

    def to_array(self) -> np.ndarray:
        features = self.price_returns + [
            self.volatility,
            self.rsi,
            self.volume_ratio,
            float(self.has_position),
            self.position_pnl_pct,
            self.account_equity_pct,
            self.drawdown,
        ]
        return np.array(features, dtype=np.float32)


@dataclass
class RLAction:
    """Discrete action space for trading."""

    HOLD = 0
    LONG = 1
    SHORT = 2
    CLOSE = 3

    @classmethod
    def size(cls) -> int:
        return 4


@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class SimpleQLearner:
    """Tabular-ish Q-Learning with function approximation.

    Uses a simple linear model for Q-value estimation.
    This is lightweight and doesn't require PyTorch.
    For full DQN/PPO, use the optional RL dependencies.
    """

    def __init__(
        self,
        state_dim: int = 17,  # 10 returns + 7 features
        n_actions: int = 4,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        memory_size: int = 10000,
        batch_size: int = 32,
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        # Simple linear Q-network (weights matrix)
        self.weights = np.random.randn(state_dim, n_actions) * 0.01
        self.bias = np.zeros(n_actions)

        # Experience replay buffer
        self.memory = deque(maxlen=memory_size)

        self.total_steps = 0
        self.total_rewards = 0.0

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        return state @ self.weights + self.bias

    def select_action(self, state: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        q_values = self.get_q_values(state)
        return int(np.argmax(q_values))

    def store_experience(self, exp: Experience):
        self.memory.append(exp)

    def train_step(self) -> float:
        """Train on a batch of experiences. Returns average loss."""
        if len(self.memory) < self.batch_size:
            return 0.0

        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in indices]

        total_loss = 0.0
        for exp in batch:
            # Current Q-values
            current_q = self.get_q_values(exp.state)

            # Target Q-value
            if exp.done:
                target = exp.reward
            else:
                next_q = self.get_q_values(exp.next_state)
                target = exp.reward + self.gamma * np.max(next_q)

            # TD error
            td_error = target - current_q[exp.action]
            total_loss += td_error**2

            # Gradient update (simple SGD)
            self.weights[:, exp.action] += self.lr * td_error * exp.state
            self.bias[exp.action] += self.lr * td_error

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.total_steps += 1

        return total_loss / self.batch_size

    def calculate_reward(
        self,
        action: int,
        pnl_change: float,
        fee: float,
        position_held: bool,
        was_liquidated: bool = False,
    ) -> float:
        """Calculate risk-adjusted reward (Sortino-inspired).

        From TensorTrade research:
        - Use asymmetric reward: losses penalized more than gains rewarded
        - This prevents the RL agent from learning high-variance strategies
        - Sortino ratio only penalizes downside volatility

        Reward design:
        - Positive PnL → reward scaled by 1.0x
        - Negative PnL → penalty scaled by 1.5x (asymmetric)
        - Fee penalty discourages excessive trading
        - Liquidation is heavily penalized
        - Drawdown-aware penalty
        """
        # Asymmetric PnL reward (Sortino-inspired)
        if pnl_change >= 0:
            reward = pnl_change * 100
        else:
            reward = pnl_change * 150  # Losses hurt 1.5x more

        reward -= fee * 10  # Fee penalty

        if was_liquidated:
            reward -= 100  # Heavy liquidation penalty (doubled)

        # Small penalty for holding with no position (opportunity cost)
        if action == RLAction.HOLD and not position_held:
            reward -= 0.01

        # Bonus for profitable close (encourages taking profits)
        if action == RLAction.CLOSE and pnl_change > 0:
            reward += pnl_change * 30  # Extra bonus for taking profits

        self.total_rewards += reward
        return reward

    def get_stats(self) -> dict:
        return {
            "epsilon": round(self.epsilon, 4),
            "total_steps": self.total_steps,
            "total_rewards": round(self.total_rewards, 2),
            "memory_size": len(self.memory),
            "avg_q": round(float(np.mean(self.weights)), 6),
        }


class RLEnhancedEvolution:
    """Combines RL with genetic evolution.

    The RL component provides:
    1. Meta-learning: learns which strategy parameters work best in which regime
    2. Position sizing: dynamically adjusts leverage and position size
    3. Exit timing: learns optimal exit points beyond fixed TP/SL

    This works alongside the genetic algorithm:
    - GA evolves strategy parameters (what to trade)
    - RL optimizes execution (when and how much)
    """

    def __init__(self, state_dim: int = 17, n_returns: int = 10):
        self.learner = SimpleQLearner(state_dim=state_dim)
        self.n_returns = n_returns
        self.returns_buffer: deque = deque(maxlen=n_returns)
        self.last_state: Optional[np.ndarray] = None
        self.last_action: Optional[int] = None

    def build_state(
        self,
        prices: list[float],
        volume: float,
        avg_volume: float,
        has_position: bool,
        position_pnl_pct: float,
        equity_pct: float,
        drawdown: float,
    ) -> RLState:
        """Build state from market data and account info."""
        # Calculate returns
        returns = []
        for i in range(1, min(len(prices), self.n_returns + 1)):
            ret = (prices[-i] - prices[-i - 1]) / prices[-i - 1] if len(prices) > i else 0
            returns.append(ret)
        while len(returns) < self.n_returns:
            returns.append(0.0)

        # Calculate RSI approximation from returns
        gains = [r for r in returns if r > 0]
        losses = [-r for r in returns if r < 0]
        avg_gain = np.mean(gains) if gains else 0
        avg_loss = np.mean(losses) if losses else 0.001
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Volatility
        vol = np.std(returns) if len(returns) > 1 else 0

        return RLState(
            price_returns=returns,
            volatility=vol,
            rsi=rsi,
            volume_ratio=volume / avg_volume if avg_volume > 0 else 1.0,
            has_position=has_position,
            position_pnl_pct=position_pnl_pct,
            account_equity_pct=equity_pct,
            drawdown=drawdown,
        )

    def decide(self, state: RLState) -> int:
        """Get action from RL agent."""
        state_array = state.to_array()
        action = self.learner.select_action(state_array)
        self.last_state = state_array
        self.last_action = action
        return action

    def feedback(
        self,
        new_state: RLState,
        pnl_change: float,
        fee: float,
        position_held: bool,
        done: bool = False,
        was_liquidated: bool = False,
    ):
        """Provide feedback to RL agent after action."""
        if self.last_state is None or self.last_action is None:
            return

        reward = self.learner.calculate_reward(
            self.last_action, pnl_change, fee, position_held, was_liquidated
        )
        new_state_array = new_state.to_array()

        exp = Experience(
            state=self.last_state,
            action=self.last_action,
            reward=reward,
            next_state=new_state_array,
            done=done,
        )
        self.learner.store_experience(exp)
        self.learner.train_step()
