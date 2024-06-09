from .agent_template import HumanoidTemplate
from .agent_eval_template import HumanoidEvalTemplate
from .agent_training_template import HumanoidTrainTemplate
from .agent_test_apg import HumanoidAPGTest
from .agent_ppo_train import  HumanoidPPOENV
from .pds_controllers_agents import *
from brax import envs
import sys
import os
from typing import Tuple

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.SimpleConverter import SimpleConverter







def register_ppo_env(args) -> HumanoidPPOENV:
        
    trajectory = SimpleConverter(args.ref)
    trajectory.load_mocap()
    model_path = 'models/final_humanoid2.xml'
    
    envs.register_environment('humanoidPPOEnv',HumanoidPPOENV)
    env_name_ppo = 'humanoidPPOEnv'
    env_ppo = generate_env_ppo(trajectory,model_path,env_name_ppo,args)

    return env_ppo






def register_mimic_env(args) -> Tuple[HumanoidTemplate,HumanoidEvalTemplate,HumanoidTrainTemplate,HumanoidAPGTest]:
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
    
    
    envs.register_environment('humanoidApgTest',HumanoidAPGTest)
    env_name_apg = 'humanoidApgTest'
    env_apg = generate_env_apg_train(trajectory,model_path,env_name_apg,args)
    
    
    
    return env_replay,env_eval,env,env_apg
    #return env_replay



def generate_env_ppo(trajectory,model_path,env_name_ppo,args):
    
    env_kwargs = dict(referece_data=trajectory,model_path=model_path,args=args) 
    
    env_ppo = envs.get_environment(env_name_ppo,**env_kwargs)
    return env_ppo
    

def generate_env_apg_train(trajectory,model_path,env_name_apg,args):
    
    #model_path = 'anybotics_anymal_c/scene_mjx.xml'
    
    env_kwargs = dict(referece_data=trajectory,model_path=model_path,args=args) 
    
    env = envs.get_environment(env_name_apg,**env_kwargs)
    return env
    



def generate_env_train(trajectory:SimpleConverter,model_path,env_name,args):
    
    env_kwargs = dict(referece_data=trajectory,model_path=model_path,args=args) 
    
    env = envs.get_environment(env_name,**env_kwargs)
    return env
    



def generate_env_replay(trajectory:SimpleConverter, model_path,env_replay_name,args):
    
    
    env_kwargs = dict(referece_data=trajectory,model_path=model_path,args=args) 
    env_replay = envs.get_environment(env_replay_name,**env_kwargs)
    return env_replay
    

def generate_env_eval(trajectory:SimpleConverter,model_path,env_eval_name,args):
    env_kwargs = dict(referece_data=trajectory,model_path=model_path,args=args) 
    
    env_eval = envs.get_environment(env_eval_name,**env_kwargs)
    return env_eval
    
    
    