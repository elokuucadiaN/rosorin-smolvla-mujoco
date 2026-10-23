"""Initial-only remote inference and exact 108-step raw action execution."""
import base64
from collections import deque
import hashlib
import io
import json
import queue
import threading
import urllib.request
import uuid
import mujoco
import numpy as np
from .factory_grasp_core import validate_model_command

WEIGHTS_SHA = '117974341c69adb4e65e47fca8de221eccd192d4cf4fe588b95665ff8e10d1b0'
STATS_SHA = '390831801e3a3ddc10c6ba37d0a9570804cdbad794ac64fe8e41313e54c081c8'

class VLAClient:
    def __init__(self,url='http://127.0.0.1:8767',timeout=60.):
        if not url.startswith('http://127.0.0.1:'):
            raise ValueError('LOCAL_SSH_BRIDGE_REQUIRED')
        self.url=url.rstrip('/');self.timeout=timeout

    def predict(self,observation,request_id):
        buffer=io.BytesIO()
        np.savez_compressed(buffer,rgb=observation['rgb'],state=observation['state'])
        request={'request_id':request_id,'initial_npz_b64':base64.b64encode(buffer.getvalue()).decode('ascii')}
        data=json.dumps(request).encode()
        message=urllib.request.Request(self.url+'/predict',data=data,headers={'Content-Type':'application/json'})
        # Ignore proxy settings: initial observations go only to the loopback SSH bridge.
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(message,timeout=self.timeout) as response:
            result=json.loads(response.read(200_000))
        if result.get('error'):raise RuntimeError(result['error'])
        return result

class PaperContactWindow:
    def __init__(self,env):
        self.dt=float(env.model.opt.timestep)
        self.count=round(2./self.dt)
        self.rows=deque(maxlen=self.count)

    def observe(self,env):
        flags=[False,False,False]
        for index,c in enumerate(env.data.contact[:env.data.ncon]):
            pair={int(c.geom1),int(c.geom2)}
            if env.block_geom not in pair:continue
            force=np.zeros(6);mujoco.mj_contactForce(env.model,env.data,index,force)
            if abs(force[0])>.01:
                for k,g in enumerate((env.left,env.right,env.paper)):flags[k]|=g in pair
        self.rows.append(tuple(flags))

    def result(self,env):
        rows=list(self.rows);n=len(rows)
        both=sum(a and b for a,b,_ in rows)
        paper=sum(p for _,_,p in rows)
        longest=gap=0
        for _,_,contact in rows:
            gap=0 if contact else gap+1;longest=max(longest,gap)
        r=env.result();reasons=[]
        if n!=self.count:reasons.append('incomplete_tail_window')
        if both!=self.count:reasons.append('bilateral_finger_contact_interrupted')
        if not n or paper/n<.95-1e-12:reasons.append('paper_contact_fraction_below_95pct')
        if longest*self.dt>.05+1e-12:reasons.append('paper_contact_gap_over_50ms')
        if not r['full_named_open_seen']:reasons.append('full_open_not_seen')
        if r['max_lift_m']>.004:reasons.append('lift_over_4mm')
        if r['arm_paper_contact_steps']:reasons.append('arm_paper_collision')
        if r['illegal_self_contact_steps']:reasons.append('illegal_self_collision')
        if r['max_mimic_error_rad']>=.005:reasons.append('mimic_error')
        if r['max_contact_penetration_m']>=.003:reasons.append('contact_penetration')
        return {'success':not reasons,'failure_reasons':reasons,
            'metric_protocol':'paper_tolerance_v1','tail_bilateral_fraction':both/n if n else 0.,
            'tail_paper_fraction':paper/n if n else 0.,'tail_longest_paper_gap_s':longest*self.dt}

class VLAExecutor:
    def __init__(self,env,client):
        self.env=env;self.client=client;self.generation=0;self.mailbox=queue.Queue()
        self.reset()

    def reset(self):
        self.generation+=1;self.state='idle';self.consumed=False;self.actions=None;self.steps=0
        self.error='';self.metadata={};self.policy_calls=0;self.final_result=None
        self.contact=PaperContactWindow(self.env)
        self.observation=self.env.initial_observation()
        self.initial_sha256=hashlib.sha256(self.observation['rgb'].tobytes()+self.observation['state'].tobytes()).hexdigest()
        self.request_id=uuid.uuid4().hex

    def begin(self):
        if self.consumed:raise ValueError('EPISODE_RESET_REQUIRED')
        if self.env.data.time!=0:raise ValueError('INITIAL_EPISODE_STATE_REQUIRED')
        self.consumed=True;self.state='inferring';self.policy_calls=1
        generation=self.generation;obs=self.observation;request_id=self.request_id
        def infer():
            try:value=self.client.predict(obs,request_id);error=None
            except Exception as exc:value=None;error=str(exc)
            self.mailbox.put((generation,value,error))
        threading.Thread(target=infer,daemon=True,name='initial_vla_request').start()

    def poll(self):
        while True:
            try:generation,result,error=self.mailbox.get_nowait()
            except queue.Empty:return
            if generation!=self.generation or self.state!='inferring':continue
            try:
                if error:raise ValueError(error)
                required={'request_id':self.request_id,'initial_sha256':self.initial_sha256,
                    'weights_sha256':WEIGHTS_SHA,'stats_sha256':STATS_SHA,'precision':'fp32',
                    'native_denoising_steps':10,'noise_seed':731,'postprocessing':'none',
                    'horizon':108,'hz':10,'initial_wait_s':0,'policy_calls':1,'input_fields':['rgb','state']}
                for name,expected in required.items():
                    if result.get(name)!=expected:raise ValueError('VLA_CONTRACT_MISMATCH: '+name)
                actions=np.asarray(result['actions'],dtype=np.float64)
                if actions.shape!=(108,6):raise ValueError('VLA_ACTION_SHAPE')
                for action in actions:validate_model_command(action)
                self.actions=actions;self.metadata={k:v for k,v in result.items() if k!='actions'}
                self.state='executing'
            except Exception as exc:
                self.error=str(exc);self.state='inference_failed';self.env.controller.stop()

    def step(self,now):
        if self.state!='executing':return
        per_action=round(.1/self.env.model.opt.timestep)
        index=self.steps//per_action
        if not self.env.controller.set_command(self.actions[index],now):
            self.error='ACTION_INVALID';self.cancel();return
        self.env.physics_step(now);self.contact.observe(self.env);self.steps+=1
        if self.steps==108*per_action:
            self.env.controller.stop();self.final_result=self.contact.result(self.env)
            self.state='succeeded' if self.final_result['success'] else 'failed'

    def cancel(self):
        self.generation+=1;self.state='cancelled';self.actions=None
        self.final_result=None
        self.env.controller.stop()

    @property
    def stage(self):
        if self.state!='executing':return ''
        index=self.steps//round(.1/self.env.model.opt.timestep)
        for end,name in ((8,'open'),(28,'approach'),(38,'descend'),(53,'close'),(108,'hold')):
            if index<end:return name
        return ''

    def result(self):
        return {**self.env.result(),**(self.final_result or {'success':False}),
            'policy':'smolvla','vla_connected':bool(self.metadata),'vla_state':self.state,
            'policy_camera_reads':self.env.policy_camera_reads,'policy_calls':self.policy_calls,
            'expert_calls':0,'postprocessing':'none','initial_wait_s':0,
            'initial_sha256':self.initial_sha256,'request_id':self.request_id,'executed_actions':self.steps//200,
            'initial_capture_sim_time_s':0.,'execution_steps':self.steps,
            'inference':self.metadata}
