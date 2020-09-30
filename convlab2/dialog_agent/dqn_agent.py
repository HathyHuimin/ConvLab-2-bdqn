"""Dialog agent interface and classes."""
from abc import ABC, abstractmethod
from convlab2.nlu import NLU
from convlab2.dst import DST
from convlab2.policy import Policy
from convlab2.nlg import NLG
from copy import deepcopy


class Agent(ABC):
    """Interface for dialog agent classes."""
    @abstractmethod
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def response(self, observation):
        pass

    @abstractmethod
    def init_session(self, **kwargs):
        pass


class PipelineAgent(Agent):

    def __init__(self, dst: DST, policy: Policy, name: str):
        super(PipelineAgent, self).__init__(name=name)
        assert self.name in ['user', 'sys']
        self.opponent_name = 'user' if self.name is 'sys' else 'sys'
        self.dst = dst
        self.policy = policy
        self.init_session()
        self.history = []

    def response(self, observation):
        """Generate agent response using the agent modules."""
        if self.dst is not None:
            self.dst.state['history'].append([self.opponent_name, observation]) # [['sys', sys_utt], ['user', user_utt],...]
        self.history.append([self.opponent_name, observation])
        # get dialog act
        self.input_action = deepcopy(observation)

        # get state
        if self.dst is not None:
            if self.name is 'sys':
                self.dst.state['user_action'] = self.input_action
            else:
                self.dst.state['system_action'] = self.input_action
            state = deepcopy(self.dst.update(self.input_action))
        else:
            state = deepcopy(self.input_action)

        # get action
        self.output_action = deepcopy(self.policy.predict(state)) # get rid of reference problem

        model_response = self.output_action

        if self.dst is not None:
            self.dst.state['history'].append([self.name, model_response])
            if self.name is 'sys':
                self.dst.state['system_action'] = self.output_action
            else:
                self.dst.state['user_action'] = self.output_action
        self.history.append([self.name, model_response])
        return model_response

    def is_terminated(self):
        if hasattr(self.policy, 'is_terminated'):
            return self.policy.is_terminated()
        return None

    def get_reward(self):
        if hasattr(self.policy, 'get_reward'):
            return self.policy.get_reward()
        return None

    def init_session(self, **kwargs):
        """Init the attributes of DST and Policy module."""
        if self.dst is not None:
            self.dst.init_session()
            if self.name == 'sys':
                self.dst.state['history'].append([self.name, 'null'])
        if self.policy is not None:
            self.policy.init_session(**kwargs)
        self.history = []

    def get_in_da(self):
        return self.input_action

    def get_out_da(self):
        return self.output_action
