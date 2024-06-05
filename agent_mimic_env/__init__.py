from .agent_template import HumanoidTemplate
from .agent_eval_template import HumanoidEvalTemplate
from .agent_training_template import HumanoidTrainTemplate
from .pds_controllers_agents import *
from brax import envs
import sys
import os
from typing import Tuple

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.SimpleConverter import SimpleConverter


def register_mimic_env(args) -> Tuple[HumanoidTemplate,HumanoidEvalTemplate,HumanoidTrainTemplate]:
#def register_mimic_env(args) -> HumanoidTemplate:
        
    trajectory = SimpleConverter(args.ref)
    trajectory.load_mocap()
    model_path = 'models/final_humanoid.xml'
    
        
    envs.register_environment('humanoidEnvReplay',HumanoidTemplate)
    env_replay_name = 'humanoidEnvReplay'
    env_replay = generate_env_replay(trajectory,model_path,env_replay_name,args)
    
    envs.register_environment('humanoidEnvEval',HumanoidEvalTemplate)
    env_eval_name = 'humanoidEnvEval'
    env_eval = generate_env_eval(trajectory,model_path,env_eval_name,args)
    
    
    envs.register_environment('humanoidEnvTrain',HumanoidTrainTemplate)
    env_name = 'humanoidEnvTrain'
    env = generate_env_train(trajectory,model_path,env_name,args)
    
     
    return env_replay,env_eval,env
    #return env_replay



def generate_env_train(trajectory:SimpleConverter,model_path,env_name,args):
    env = envs.get_environment(env_name=env_name,
                           reference_data=trajectory,
                           model_path=model_path,
                           args=args)
    return env
    



def generate_env_replay(trajectory:SimpleConverter, model_path,env_replay_name,args):
    env_replay = envs.get_environment(env_name=env_replay_name,
                           reference_data=trajectory,
                           model_path=model_path,
                           args=args)
    return env_replay
    

def generate_env_eval(trajectory:SimpleConverter,model_path,env_eval_name,args):
    env_eval = envs.get_environment(env_name=env_eval_name,
                           reference_data=trajectory,
                           model_path=model_path,
                           args=args)
    return env_eval
    
    
    