# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
from abc import ABC, abstractmethod
from collections import defaultdict
from os import getcwd
from random import choices
from typing import Dict

from maro.communication import Proxy, SessionType
from maro.rl.experience import ExperienceSet
from maro.rl.exploration import AbsExploration
from maro.rl.env_wrapper import AbsEnvWrapper
from maro.rl.policy import AbsPolicy
from maro.utils import Logger

from .message_enums import MsgTag, MsgKey


class AbsRolloutManager(ABC):
    """Learner class for distributed training.

    Args:
        agent (Union[AbsPolicy, MultiAgentWrapper]): Learning agents.
        scheduler (Scheduler): .
        num_actors (int): Expected number of actors in the group identified by ``group_name``.
        group_name (str): Identifier of the group to which the actor belongs. It must be the same group name
            assigned to the actors (and roll-out clients, if any).
        proxy_kwargs: Keyword parameters for the internal ``Proxy`` instance. See ``Proxy`` class
            for details. Defaults to None.
        update_trigger (str): Number or percentage of ``MsgTag.ROLLOUT_DONE`` messages required to trigger
            learner updates, i.e., model training.
    """
    def __init__(self):
        super().__init__()
        self.episode_complete = False

    @abstractmethod
    def collect(self, ep: int, segment: int, policy_state_dict: dict):
        """Collect experiences from actors."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, policy_dict: dict):
        raise NotImplementedError

    def reset(self):
        self.episode_complete = False


class LocalRolloutManager(AbsRolloutManager):
    def __init__(
        self,
        env: AbsEnvWrapper,
        policy_dict: Dict[str, AbsPolicy],
        agent2policy: Dict[str, str],
        exploration_dict: Dict[str, AbsExploration] = None,
        agent2exploration: Dict[str, str] = None,
        num_steps: int = -1,
        eval_env: AbsEnvWrapper = None,
        log_env_metrics: bool = True,
        log_total_reward: bool = True,
        log_dir: str = getcwd(),
    ):
        if num_steps == 0 or num_steps < -1:
            raise ValueError("num_steps must be a positive integer or -1")

        self._logger = Logger("LOCAL_ROLLOUT_MANAGER", dump_folder=log_dir)

        self.env = env
        self.eval_env = eval_env if eval_env else self.env
        
        # mappings between agents and policies
        self.policy_dict = policy_dict
        self._agent2policy = agent2policy
        self._policy = {agent_id: policy_dict[policy_id] for agent_id, policy_id in self._agent2policy.items()}
        self._agent_groups_by_policy = defaultdict(list)
        for agent_id, policy_id in agent2policy.items():
            self._agent_groups_by_policy[policy_id].append(agent_id)

        # mappings between exploration schemes and agents
        self.exploration_dict = exploration_dict
        if exploration_dict:
            self._agent2exploration = agent2exploration
            self._exploration = {
                agent_id: self.exploration_dict[exploration_id]
                for agent_id, exploration_id in self._agent2exploration.items()
            }
            self._agent_groups_by_exploration = defaultdict(list)
            for agent_id, exploration_id in self._agent2exploration.items():
                self._agent_groups_by_exploration[exploration_id].append(agent_id)

        self._num_steps = num_steps if num_steps > 0 else float("inf")
        self._log_env_metrics = log_env_metrics
        self._log_total_reward = log_total_reward
        self._eval_ep = 0

    def collect(self, ep: int, segment: int, policy_state_dict: dict):
        t0 = time.time()
        learning_time = 0
        num_experiences_collected = 0

        if self.exploration_dict:
            exploration_params = {
                tuple(agent_ids): self.exploration_dict[exploration_id].parameters
                for exploration_id, agent_ids in self._agent_groups_by_exploration.items()
            }
            self._logger.debug(f"Exploration parameters: {exploration_params}")

        if self.env.state is None:
            self._logger.info(f"Collecting data from episode {ep}, segment {segment}")
            if hasattr(self, "exploration_dict"):
                exploration_params = {
                    tuple(agent_ids): self.exploration_dict[exploration_id].parameters
                    for exploration_id, agent_ids in self._agent_groups_by_exploration.items()
                }
                self._logger.debug(f"Exploration parameters: {exploration_params}")

            self.env.reset()
            self.env.start()  # get initial state

        # load policies
        self._load_policy_states(policy_state_dict)

        start_step_index = self.env.step_index + 1
        steps_to_go = self._num_steps
        while self.env.state and steps_to_go > 0:
            if self.exploration_dict:      
                action = {
                    id_:
                        self._exploration[id_](self._policy[id_].choose_action(st))
                        if id_ in self._exploration else self._policy[id_].choose_action(st)
                    for id_, st in self.env.state.items()
                }
            else:
                action = {id_: self._policy[id_].choose_action(st) for id_, st in self.env.state.items()}

            self.env.step(action)
            steps_to_go -= 1

        self._logger.info(
            f"Roll-out finished for ep {ep}, segment {segment}"
            f"(steps {start_step_index} - {self.env.step_index})"
        )

        # update the exploration parameters if an episode is finished
        if not self.env.state and self.exploration_dict:
            for exploration in self.exploration_dict.values():
                exploration.step()

        # performance details
        if not self.env.state:
            if self._log_env_metrics:
                self._logger.info(f"ep {ep}: {self.env.metrics}")
            if self._log_total_reward:
                self._logger.info(f"ep {ep} total reward received: {self.env.total_reward}")

            self._logger.debug(
                f"ep {ep} summary - "
                f"running time: {time.time() - t0}"
                f"env steps: {self.env.step_index}"    
                f"learning time: {learning_time}"
                f"experiences collected: {num_experiences_collected}"
            )

        return self.env.get_experiences()

    def evaluate(self, policy_state_dict: dict):
        self._logger.info("Evaluating...")
        self._eval_ep += 1
        self._load_policy_states(policy_state_dict)
        self.eval_env.save_replay = False
        self.eval_env.reset()
        self.eval_env.start()  # get initial state
        while self.eval_env.state:
            action = {id_: self._policy[id_].choose_action(st) for id_, st in self.eval_env.state.items()}
            self.eval_env.step(action)

        if not self.eval_env.state:
            self._logger.info(f"total reward: {self.eval_env.total_reward}")

        if self._log_env_metrics:
            self._logger.info(f"eval ep {self._eval_ep}: {self.eval_env.metrics}")
    
    def _load_policy_states(self, policy_state_dict: dict):
        for policy_id, policy_state in policy_state_dict.items():
            self.policy_dict[policy_id].set_state(policy_state)

        if policy_state_dict:
            self._logger.info(f"updated policies {list(policy_state_dict.keys())}")


class ParallelRolloutManager(AbsRolloutManager):
    """Learner class for distributed training.

    Args:
        agent (Union[AbsPolicy, MultiAgentWrapper]): Learning agents.
        num_actors (int): Expected number of actors in the group identified by ``group_name``.
        group_name (str): Identifier of the group to which the actor belongs. It must be the same group name
            assigned to the actors (and roll-out clients, if any).
        proxy_kwargs: Keyword parameters for the internal ``Proxy`` instance. See ``Proxy`` class
            for details. Defaults to None.
        update_trigger (str): Number or percentage of ``MsgTag.ROLLOUT_DONE`` messages required to trigger
            learner updates, i.e., model training.
    """
    def __init__(
        self,
        num_actors: int,
        group_name: str,
        num_steps: int,
        required_finishes: int = None,
        max_staleness: int = 0,
        num_eval_actors: int = 1,
        log_env_metrics: bool = False,
        log_dir: str = getcwd(),
        **proxy_kwargs
    ):
        super().__init__()
        if required_finishes and required_finishes > num_actors:
            raise ValueError("required_finishes cannot exceed the number of available actors")
        if num_eval_actors > num_actors:
            raise ValueError("num_eval_actors cannot exceed the number of available actors")

        self._logger = Logger("PARALLEL_ROLLOUT_MANAGER", dump_folder=log_dir)
        self.num_actors = num_actors
        peers = {"actor": num_actors}
        self._proxy = Proxy(group_name, "actor_manager", peers, **proxy_kwargs)
        self._actors = self._proxy.peers["actor"]  # remote actor ID's

        if required_finishes is None:
            required_finishes = self.num_actors
            self._logger.info(f"Required number of actor finishes is set to {required_finishes}")

        self.required_finishes = required_finishes
        self._num_steps = num_steps
        self._max_staleness = max_staleness
        self.total_experiences_collected = 0
        self.total_env_steps = 0
        self.total_reward = defaultdict(float)
        self._log_env_metrics = log_env_metrics

        self._num_eval_actors = num_eval_actors
        self._eval_ep = 0

    def collect(
        self,
        episode_index: int,
        segment_index: int,
        policy_state_dict: dict
    ):
        """Collect experiences from actors."""
        msg_body = {
            MsgKey.EPISODE_INDEX: episode_index,
            MsgKey.SEGMENT_INDEX: segment_index,
            MsgKey.NUM_STEPS: self._num_steps,
            MsgKey.POLICY: policy_state_dict,
            MsgKey.RETURN_ENV_METRICS: self._log_env_metrics
        }

        if self._log_env_metrics:
            self._logger.info(f"EPISODE-{episode_index}, SEGMENT-{segment_index}: ")

        self._proxy.ibroadcast("actor", MsgTag.COLLECT, SessionType.TASK, body=msg_body)
        self._logger.info(f"Sent collect requests for ep-{episode_index}, segment-{segment_index}")

        # Receive roll-out results from remote actors
        combined_exp_by_agent = defaultdict(ExperienceSet)
        num_segment_finishes = num_episode_finishes = 0
        for msg in self._proxy.receive():
            if msg.tag != MsgTag.COLLECT_DONE or msg.body[MsgKey.EPISODE_INDEX] != episode_index:
                self._logger.info(
                    f"Ignore a message of type {msg.tag} with roll-out index {msg.body[MsgKey.EPISODE_INDEX]} "
                    f"(expected message type {MsgTag.COLLECT} and roll-out index {episode_index})"
                )
                continue

            # log roll-out summary
            if self._log_env_metrics:
                env_metrics = msg.body[MsgKey.METRICS]
                self._logger.info(f"env_metrics: {env_metrics}")

            if segment_index - msg.body[MsgKey.SEGMENT_INDEX] <= self._max_staleness:
                exp_by_agent = msg.body[MsgKey.EXPERIENCES]
                self.total_experiences_collected += sum(exp.size for exp in exp_by_agent.values())
                self.total_env_steps += msg.body[MsgKey.NUM_STEPS]
                is_episode_end = msg.body[MsgKey.EPISODE_END]
                if is_episode_end:
                    self._logger.info(f"total rewards: {msg.body[MsgKey.TOTAL_REWARD]}")
                num_episode_finishes += is_episode_end
                if num_episode_finishes == self.required_finishes:
                    self.episode_complete = True

                for agent_id, exp in exp_by_agent.items():
                    combined_exp_by_agent[agent_id].extend(exp)

            if msg.body[MsgKey.SEGMENT_INDEX] == segment_index:
                num_segment_finishes += 1
                if num_segment_finishes == self.required_finishes:
                    break

        return combined_exp_by_agent

    def evaluate(self, policy_state_dict: dict):
        """Evaluate ."""
        self._eval_ep += 1
        msg_body = {
            MsgKey.EPISODE_INDEX: self._eval_ep,
            MsgKey.POLICY: policy_state_dict,
            MsgKey.RETURN_ENV_METRICS: True
        }

        actors = choices(self._actors, k=self._num_eval_actors)
        self._proxy.iscatter(MsgTag.EVAL, SessionType.TASK, [(actor_id, msg_body) for actor_id in actors])
        self._logger.info(f"Sent evaluation requests to {actors}")

        # Receive roll-out results from remote actors
        num_finishes = 0
        for msg in self._proxy.receive():
            if msg.tag != MsgTag.EVAL_DONE or msg.body[MsgKey.EPISODE_INDEX] != self._eval_ep:
                self._logger.info(
                    f"Ignore a message of type {msg.tag} with episode index {msg.body[MsgKey.EPISODE_INDEX]} "
                    f"(expected message type {MsgTag.EVAL} and episode index {self._eval_ep})"
                )
                continue

            # log roll-out summary
            env_metrics = msg.body[MsgKey.METRICS]
            self._logger.info(f"env metrics for evaluation episode {self._eval_ep}: {env_metrics}")

            if msg.body[MsgKey.EPISODE_INDEX] == self._eval_ep:
                num_finishes += 1
                if num_finishes == self._num_eval_actors:
                    break

    def exit(self):
        """Tell the remote actors to exit."""
        self._proxy.ibroadcast("actor", MsgTag.EXIT, SessionType.NOTIFICATION)
        self._logger.info("Exiting...")
