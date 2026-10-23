"""Factory clamp core; regenerate with tools/migrate_factory_runtime.py.

No pad geometry, camera aim offset, object weld or runtime object teleport.
"""
import json, math
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from scipy.optimize import least_squares
from .factory_kinematics import fk, set_state, mesh_vertices_world

ARM = tuple(f'joint{i}' for i in range(1,6))

def validate_model_command(command):
    values = np.asarray(command, dtype=float)
    if (values.shape != (6,) or not np.isfinite(values).all()
            or np.any(np.abs(values[:5]) > 2.09) or abs(values[5]) > 1.57):
        raise ValueError('SIX_FACTORY_JOINT_ANGLES_IN_RADIANS_REQUIRED')
    return values

SOURCE_PROTOTYPE_SHA256 = '3c6dd320f41d04740df2c44f5136f759aaa4d1cf4ff322dae81a1d4a30540a79'

class FactoryGraspCore:
    def __init__(self, scene, assets):
        self.path = Path(scene)
        self.model = mujoco.MjModel.from_xml_path(str(self.path))
        self.data = mujoco.MjData(self.model)
        self.joints = {j.find('child').get('link'): j for j in ET.parse(Path(assets)/'rosorin_factory.urdf').getroot().findall('joint')}
        self.validation = json.loads((Path(assets)/'validation.json').read_text(encoding='utf-8'))
        self.initial = np.array([self.validation['selected_task_initial_reference']['joint_angles_rad'][n] for n in ARM]+[0.])
        self.qadr = np.array([self.model.jnt_qposadr[self.model.joint(n).id] for n in ARM+('r_joint',)])
        self.tcp_cache = {}
        self.block_qadr = self.model.jnt_qposadr[self.model.joint('red_block_free').id]
        self.block = self.model.body('red_block').id
        self.block_geom = self.model.geom('red_block_geom').id
        self.paper = self.model.geom('white_paper').id
        self.left = self.model.geom('factory_l_out_link').id
        self.right = self.model.geom('factory_r_out_link').id
        FactoryGraspCore.reset(self, (.212, .035))

    def reset(self,xy=(.26,.035)):
        mujoco.mj_resetData(self.model,self.data)
        self.reference=self.initial.copy()
        set_state(self.model,self.data,self.joints,dict(zip(ARM+('r_joint',),self.initial)))
        self.data.qpos[self.block_qadr:self.block_qadr+7]=[*xy,.0905,1,0,0,0]
        self.data.ctrl[:]=self.initial
        mujoco.mj_forward(self.model,self.data)
        self.max_height=.0905;self.open_seen=False;self.hold_streak=0.;self.max_hold=0.;self.arm_paper_contacts=0;self.finger_paper_contacts=0
        self.max_mimic_error=0.;self.max_contact_penetration=0.;self.illegal_self_contacts=0;self.self_collision_pairs=set()
        self.object_initial=np.array([*xy,.0905])
        self.frozen_observation=None;self.policy_camera_reads=0

    def initial_observation(self):
        if self.frozen_observation is None:
            if self.data.time!=0:
                raise RuntimeError('INITIAL_OBSERVATION_MUST_BE_CAPTURED_BEFORE_ACTIONS')
            with mujoco.Renderer(self.model,height=400,width=640) as renderer:
                renderer.update_scene(self.data,camera='factory_rgb')
                rgb=renderer.render().copy()
            state=self.data.qpos[self.qadr].astype(np.float32).copy()
            rgb.setflags(write=False);state.setflags(write=False)
            self.frozen_observation={'rgb':rgb,'state':state};self.policy_camera_reads+=1
        return self.frozen_observation

    def expert_plan(self):
        xy=self.object_initial[:2]
        approach,_=self.solve(np.r_[xy,.16],self.initial[:5],opening=.82)
        grasp_seed,_=self.solve(np.r_[xy,.0925],approach,opening=.82)
        close,_=self.solve(np.r_[xy,.0925],grasp_seed,opening=-.2)
        # The open/closed finger tips differ by about 24mm along the grasp axis.
        # Descend to the FINAL arm posture, then close only the fingers. Executing
        # grasp_seed as an open-gripper descent caused the unwanted dip/rebound.
        return [('wait',self.initial.copy(),5.),('open',np.r_[self.initial[:5],.82],.8),
            ('approach',np.r_[approach,.82],2.),('descend',np.r_[close,.82],1.),
            ('close',np.r_[close,-.68],1.5),('hold',np.r_[close,-.68],5.5)]

    @staticmethod
    def smooth_reference(start,target,progress):
        u=float(np.clip(progress,0.,1.))
        return np.asarray(start)+(np.asarray(target)-np.asarray(start))*(u**3*(10+u*(-15+6*u)))

    def command(self,target,duration,*,smooth=False):
        target=np.array(validate_model_command(target));dt=self.model.opt.timestep
        start=self.reference.copy();steps=round(duration/dt)
        for step in range(steps):
            desired=self.smooth_reference(start,target,(step+1)/steps) if smooth else target
            self.reference+=np.clip(desired-self.reference,-2.*dt,2.*dt)
            self.data.ctrl[:]=self.reference
            mujoco.mj_step(self.model,self.data)
            self.update_metrics()

    def update_metrics(self):
        dt=self.model.opt.timestep
        angle=float(self.data.qpos[self.qadr[-1]])
        self.open_seen|=abs(angle-.82)<.025
        for name,mimic in self.validation['gripper_mimic_map'].items():
            q=self.data.qpos[self.model.jnt_qposadr[self.model.joint(name).id]]
            self.max_mimic_error=max(self.max_mimic_error,abs(q-angle*mimic['multiplier']-mimic['offset']))
        sides=set();on_paper=False
        for contact_index,contact in enumerate(self.data.contact[:self.data.ncon]):
            pair={int(contact.geom1),int(contact.geom2)}
            names=[mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,g) or '' for g in pair]
            if len(names)==2 and all(name.startswith('factory_') for name in names) and contact.dist<-.0005:
                self.illegal_self_contacts+=1;self.self_collision_pairs.add(tuple(sorted(names)))
            self.max_contact_penetration=max(self.max_contact_penetration,-float(contact.dist))
            if self.block_geom in pair:
                force=np.zeros(6);mujoco.mj_contactForce(self.model,self.data,contact_index,force)
                if abs(force[0])>.01:
                    sides.update(pair.intersection({self.left,self.right}))
                    on_paper|=self.paper in pair
            if self.paper in pair:
                robot_geoms=[g for g in pair if int(self.model.geom_bodyid[g]) not in (0,self.block)]
                for geom in robot_geoms:
                    name=mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_BODY,int(self.model.geom_bodyid[geom]))
                    if name in ('l_link','r_link','l_in_link','l_out_link','r_in_link','r_out_link'):
                        self.finger_paper_contacts+=1
                    else:
                        self.arm_paper_contacts+=1
        held=self.open_seen and len(sides)==2 and on_paper
        self.hold_streak=self.hold_streak+dt if held else 0.;self.max_hold=max(self.max_hold,self.hold_streak)
        self.max_height=max(self.max_height,float(self.data.xpos[self.block,2]))

    def geometric_grasp_center(self,q):
        angles=dict(zip(ARM+('r_joint',),q))
        gripper=fk(self.joints,'gripper_link',angles)
        return gripper[:3,3]+gripper[:3,:3]@self.local_contact_center(float(q[-1]))

    def local_contact_center(self,opening):
        # Task TCP is 8mm proximal to the tips of the actual mesh, suitable for
        # the 20mm block. It adds no pad, collision shape or robot link.
        key=round(opening,8)
        if key not in self.tcp_cache:
            angles={name:0. for name in ARM};angles['r_joint']=opening
            scratch=mujoco.MjData(self.model);set_state(self.model,scratch,self.joints,angles)
            gripper=fk(self.joints,'gripper_link',angles);points=[]
            for name in ('l_out_link','r_out_link'):
                mesh=(mesh_vertices_world(self.model,scratch,name)-gripper[:3,3])@gripper[:3,:3]
                tip=mesh[:,0].max();distal=mesh[mesh[:,0]>tip-.010]
                center=(distal.min(axis=0)+distal.max(axis=0))/2
                center[0]=tip-.008;points.append(center)
            self.tcp_cache[key]=np.mean(points,axis=0)
        return self.tcp_cache[key]

    def solve(self,target,seed,opening=-.2):
        tcp=self.local_contact_center(opening)
        def residual(arm):
            angles=dict(zip(ARM,arm));angles['r_joint']=opening
            # Only geometric FK needed here; no physics state is teleported during execution.
            gripper=fk(self.joints,'gripper_link',angles)
            center=gripper[:3,3]+gripper[:3,:3]@tcp
            return np.r_[(center-target)*10,(gripper[:3,0]-[0,0,-1])*.3,arm[4]*.01]
        result=least_squares(residual,seed,bounds=([-2.09]*5,[2.09]*5),max_nfev=120)
        center=self.geometric_grasp_center(np.r_[result.x,opening])
        error=float(np.linalg.norm(center-target))
        if error>.003:
            raise ValueError(f'GRASP_IK_UNREACHABLE: {error}')
        return result.x,error

    def initial_visible_pixels(self):
        with mujoco.Renderer(self.model,height=400,width=640) as renderer:
            renderer.enable_segmentation_rendering();renderer.update_scene(self.data,camera='factory_rgb')
            image=renderer.render()
            return int(np.count_nonzero((image[:,:,0]==self.block_geom)&(image[:,:,1]==mujoco.mjtObj.mjOBJ_GEOM)))

    def result(self):
        success=(self.open_seen and self.max_hold>=2 and self.hold_streak>=2
            and self.max_height-.0905<=.004 and self.arm_paper_contacts==0
            and self.illegal_self_contacts==0 and self.max_mimic_error<.005
            and self.max_contact_penetration<.003)
        return {'success':bool(success),'max_continuous_clamp_on_paper_s':self.max_hold,'max_lift_m':self.max_height-.0905,
            'full_named_open_seen':bool(self.open_seen),'arm_paper_contact_steps':self.arm_paper_contacts,
            'finger_paper_contact_steps':self.finger_paper_contacts,
            'illegal_self_contact_steps':self.illegal_self_contacts,'self_collision_pairs':sorted(self.self_collision_pairs),
            'max_mimic_error_rad':float(self.max_mimic_error),'max_contact_penetration_m':self.max_contact_penetration,
            'block_final_position_m':self.data.xpos[self.block].tolist(),
            'simulation_time_s':float(self.data.time),'real_robot_ready':False}
