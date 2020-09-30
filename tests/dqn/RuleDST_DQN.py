from convlab2.dst.rule.multiwoz import RuleDST
from convlab2.policy.rule.multiwoz import RulePolicy
from convlab2.policy.ppo.multiwoz import PPOPolicy
from convlab2.policy.dqn import DQNPolicy
from convlab2.dialog_agent import PipelineAgent, BiSession
from convlab2.util.analysis_tool.analyzer import Analyzer

import random
import numpy as np
import torch


def set_seed(r_seed):
    random.seed(r_seed)
    np.random.seed(r_seed)
    torch.manual_seed(r_seed)


def test_end2end():
    sys_dst = RuleDST()
    sys_policy = DQNPolicy()
    sys_agent = PipelineAgent(None, sys_dst, sys_policy, None, name='sys')

    user_policy = RulePolicy(character='usr')
    user_agent = PipelineAgent(None, None, user_policy, None, name='user')

    analyzer = Analyzer(user_agent=user_agent, dataset='multiwoz')

    set_seed(20200202)
    analyzer.comprehensive_analyze(sys_agent=sys_agent, model_name='RuleDST-DQNPolicy', total_dialog=1000)


if __name__ == '__main__':
    test_end2end()
