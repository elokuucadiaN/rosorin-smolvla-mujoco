"""ROS-compatible factory nominal clamp fixture, with absolute radian actions."""
import math
from pathlib import Path
import mujoco
import numpy as np
from .arm_control import ArmController
from .factory_grasp_core import FactoryGraspCore, validate_model_command

COMMAND_JOINTS = tuple(f'joint{i}' for i in range(1, 6)) + ('r_joint',)
ACCEPTANCE_POSITIONS = tuple((x, y) for x in (.205, .212, .219) for y in (.025, .045))


class FactoryArmController(ArmController):
    def __init__(self, model, data):
        super().__init__(model, data)
        self.rate[:] = 2.

    def state(self):
        return self.data.qpos[self.qadr].copy()

    def set_command(self, action, now):
        try:
            values = validate_model_command(action)
            if not math.isfinite(now):
                raise ValueError('INVALID_TIME')
        except (TypeError, ValueError, OverflowError):
            self.stop()
            return False
        self.target = values.copy()
        self.last_command = now
        self.timed_out = False
        return True


def package_share():
    local = Path(__file__).resolve().parents[1]
    if (local/'worlds/manipulation_room.xml').exists():
        return local
    from ament_index_python.packages import get_package_share_directory
    return Path(get_package_share_directory('rosorin_mujoco_sim'))


class ManipulationEnv(FactoryGraspCore):
    def __init__(self, scene=None):
        share = package_share()
        super().__init__(scene or share/'worlds/manipulation_room.xml', share/'assets/factory_baseline')
        self.arm_qadr = self.qadr[:5]
        self.reset()

    def reset(self, seed=0, *, xy=None):
        self.seed = int(seed)
        position = ACCEPTANCE_POSITIONS[self.seed % len(ACCEPTANCE_POSITIONS)] if xy is None else xy
        FactoryGraspCore.reset(self, position)
        self.controller = FactoryArmController(self.model, self.data)
        self.reference = self.controller.reference
        self.trace = []

    def observe(self):
        return self.controller.state()

    def physics_step(self, command_time):
        self.controller.apply(self.model.opt.timestep, command_time)
        mujoco.mj_step(self.model, self.data)
        self.update_metrics()

    def command(self, target, duration, *, smooth=False, callback=None):
        target=validate_model_command(target)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('POSITIVE_DURATION_REQUIRED')
        start=self.controller.reference.copy()
        dt=self.model.opt.timestep;steps=round(duration/dt)
        for step in range(steps):
            desired=self.smooth_reference(start,target,(step+1)/steps) if smooth else target
            self.controller.set_command(desired, float(self.data.time))
            self.physics_step(float(self.data.time))
            if callback is not None and ((step+1)%round(.1/dt)==0 or step+1==steps):
                callback()

    def step(self, action, duration=.2):
        self.command(action, duration)
        return self.observe()

    def expert(self, callback=None):
        self.initial_observation()
        for name, action, duration in self.expert_plan():
            self.command(action,duration,smooth=True,
                         callback=(lambda:callback(name,self)) if callback is not None else None)
            self.trace.append({'stage':name, 'time_s':float(self.data.time), **self.result()})
        self.controller.stop()
        return self.result()

    def result(self):
        return {**super().result(), 'profile':'factory_nominal_simulation',
                'task':'open_approach_close_hold_on_paper', 'command_units':'radians',
                'seed':getattr(self, 'seed', 0), 'vla_connected':False}
