from maro.rl_v3 import run_workflow
from maro.rl_v3.learning import SimpleAgentWrapper, SimpleTrainerManager
from maro.simulator import Env
from .callbacks import cim_post_collect, cim_post_evaluate
from .config import algorithm, env_conf
from .env_sampler import CIMEnvSampler
from .policies import get_policy_func_dict
from .trainers import get_trainer_func_dict, policy2trainer

if __name__ == "__main__":
    run_workflow(
        get_env_sampler_func=lambda: CIMEnvSampler(
            get_env_func=lambda: Env(**env_conf),
            get_policy_func_dict=get_policy_func_dict,
            agent2policy={agent: f"{algorithm}.{agent}" for agent in Env(**env_conf).agent_idx_list},
            agent_wrapper_cls=SimpleAgentWrapper,
        ),
        get_trainer_manager_func=lambda: SimpleTrainerManager(
            get_trainer_func_dict=get_trainer_func_dict,
            get_policy_func_dict=get_policy_func_dict,
            agent2policy={agent: f"{algorithm}.{agent}" for agent in Env(**env_conf).agent_idx_list},
            policy2trainer=policy2trainer
        ),
        num_episodes=30,
        post_collect=cim_post_collect,
        post_evaluate=cim_post_evaluate
    )