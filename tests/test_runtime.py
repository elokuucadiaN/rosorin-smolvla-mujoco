"""Check protocol failures and reset behavior without substituting a fake grasp."""
import hashlib
import threading
import time
import numpy as np
import pytest
from rosorin_mujoco_sim.factory_manipulation_env import ManipulationEnv
from rosorin_mujoco_sim.vla_execution import VLAExecutor,WEIGHTS_SHA,STATS_SHA

def reply(obs,request_id):
    return {'request_id':request_id,'initial_sha256':hashlib.sha256(obs['rgb'].tobytes()+obs['state'].tobytes()).hexdigest(),
            'weights_sha256':WEIGHTS_SHA,'stats_sha256':STATS_SHA,'precision':'fp32','native_denoising_steps':10,
            'noise_seed':731,'postprocessing':'none','horizon':108,'hz':10,'initial_wait_s':0,
            'policy_calls':1,'input_fields':['rgb','state'],'actions':np.zeros((108,6)).tolist()}

@pytest.mark.parametrize('problem',['weights','shape','nan','limit'])
def test_invalid_prediction_never_moves(problem):
    class Client:
        def predict(self,obs,request_id):
            r=reply(obs,request_id)
            if problem=='weights':r['weights_sha256']='wrong'
            if problem=='shape':r['actions'].pop()
            if problem=='nan':r['actions'][0][0]=float('nan')
            if problem=='limit':r['actions'][0][0]=3.
            return r
    env=ManipulationEnv();executor=VLAExecutor(env,Client());initial=env.data.qpos.copy()
    executor.begin()
    end=time.monotonic()+5
    while executor.state=='inferring' and time.monotonic()<end:executor.poll();time.sleep(.01)
    assert executor.state=='inference_failed' and executor.steps==0 and env.data.time==0
    np.testing.assert_array_equal(initial,env.data.qpos)

def test_reset_discards_inflight_prediction():
    entered=threading.Event();finish=threading.Event()
    class Client:
        def predict(self,obs,request_id):
            entered.set();finish.wait(5);return reply(obs,request_id)
    env=ManipulationEnv();executor=VLAExecutor(env,Client());executor.begin();assert entered.wait(5)
    executor.cancel();env.reset(1);executor.reset();finish.set();time.sleep(.05);executor.poll()
    assert executor.state=='idle' and executor.steps==0 and env.data.time==0
    assert env.policy_camera_reads==1
