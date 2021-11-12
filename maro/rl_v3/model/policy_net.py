from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple

import torch.nn
from torch.distributions import Categorical

from maro.rl_v3.model import AbsNet
from maro.rl_v3.utils import match_shape
from maro.rl_v3.utils.objects import SHAPE_CHECK_FLAG


class PolicyNet(AbsNet, metaclass=ABCMeta):
    """
    Base class for all nets that serve as policy core. It has the concept of 'state' and 'action'.
    """
    def __init__(self, state_dim: int, action_dim: int) -> None:
        """
        Args:
            state_dim (int): Dimension of states.
            action_dim (int): Dimension of actions.
        """
        super(PolicyNet, self).__init__()
        self._state_dim = state_dim
        self._action_dim = action_dim

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def get_actions_with_logps(
        self, states: torch.Tensor, exploring: bool, require_logps: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get actions, and log probabilities of the actions if required, according to the states.

        Args:
            states (torch.Tensor): States.
            exploring (bool): If it is True, return the results under exploring mode. Otherwise, return the results
                under exploiting mode.
            require_logps (bool): If the return value should contains log probabilities. Defaults to True.

        Returns:
            Actions
            Log probabilities if require_logps == True else None.
        """
        assert self._shape_check(states=states), \
            f"States shape check failed. Expecting: {('BATCH_SIZE', self.state_dim)}, actual: {states.shape}."
        actions, logps = self._get_actions_with_logps_impl(states, exploring, require_logps)
        assert self._shape_check(states=states, actions=actions), \
            f"Actions shape check failed. Expecting: {(states.shape[0], self.action_dim)}, actual: {actions.shape}."
        return actions, logps

    def _shape_check(self, states: torch.Tensor, actions: Optional[torch.Tensor] = None) -> bool:
        if not SHAPE_CHECK_FLAG:
            return True
        else:
            if states.shape[0] == 0:
                return False
            if not match_shape(states, (None, self.state_dim)):
                return False

            if actions is not None:
                if not match_shape(actions, (states.shape[0], self.action_dim)):
                    return False
            return True

    @abstractmethod
    def _get_actions_with_logps_impl(
        self, states: torch.Tensor, exploring: bool, require_logps: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Implementation of `get_actions_with_logps`.
        """
        raise NotImplementedError


class DiscretePolicyNet(PolicyNet, metaclass=ABCMeta):
    """
    Net for discrete policies.
    """
    def __init__(self, state_dim: int, action_num: int) -> None:
        """
        Args:
            state_dim (int): Dimension of states.
            action_num (int): Number of actions.
        """
        super(DiscretePolicyNet, self).__init__(state_dim=state_dim, action_dim=1)
        self._action_num = action_num

    @property
    def action_num(self) -> int:
        return self._action_num

    def get_action_probs(self, states: torch.Tensor) -> torch.Tensor:
        """
        Get the probabilities for all actions.

        Args:
            states (torch.Tensor): States.

        Returns:
            Probability matrix with shape [batch_size, action_num]
        """
        assert self._shape_check(states=states), \
            f"States shape check failed. Expecting: {('BATCH_SIZE', self.state_dim)}, actual: {states.shape}."
        action_probs = self._get_action_probs_impl(states)
        assert match_shape(action_probs, (states.shape[0], self.action_num)), \
            f"Action probabilities shape check failed. Expecting: {(states.shape[0], self.action_num)}, " \
            f"actual: {action_probs.shape}."
        return action_probs

    def get_action_logps(self, states: torch.Tensor) -> torch.Tensor:
        """
        Get the log-probabilities for all actions.

        Args:
            states (torch.Tensor): States.

        Returns:
            Lop-probability matrix with shape [batch_size, action_num]
        """
        return torch.log(self.get_action_probs(states))

    @abstractmethod
    def _get_action_probs_impl(self, states: torch.Tensor) -> torch.Tensor:
        """
        Implementation of `get_action_probs`.
        """
        raise NotImplementedError

    def _get_actions_with_logps_impl(
        self, states: torch.Tensor, exploring: bool, require_logps: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if exploring:
            actions, logps = self._get_actions_exploring_impl(states, require_logps)
            return actions, logps
        else:
            action_logps = self.get_action_logps(states)
            logps, actions = action_logps.max(dim=1)
            return actions.unsqueeze(1), logps if require_logps else None

    def _get_actions_exploring_impl(
        self, states: torch.Tensor, require_logps: bool
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get actions, and log probabilities of the actions if required, according to the states under exploring mode.

        Args:
            states (torch.Tensor): States.
            require_logps (bool): If the return value should contains log probabilities. Defaults to True.

        Returns:
            Actions
            Log probabilities if require_logps == True else None.
        """
        action_probs = Categorical(self.get_action_probs(states))
        actions = action_probs.sample()
        if require_logps:
            logps = action_probs.log_prob(actions)
            return actions.unsqueeze(1), logps
        else:
            return actions.unsqueeze(1), None


class ContinuousPolicyNet(PolicyNet, metaclass=ABCMeta):
    """
    Net for continuous policies.
    """
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super(ContinuousPolicyNet, self).__init__(state_dim=state_dim, action_dim=action_dim)