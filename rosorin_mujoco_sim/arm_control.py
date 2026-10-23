"""Six-channel, rate-limited actuator control; no pose teleporting."""
import math
import mujoco
import numpy as np

ARM_JOINTS = tuple(f'joint{i}' for i in range(1, 6))
COMMAND_JOINTS = ARM_JOINTS + ('gripper_openness',)


class ArmController:
    def __init__(self, model, data, timeout=.5):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('timeout must be finite and positive')
        self.model, self.data, self.timeout = model, data, timeout
        self.qadr = np.array([model.jnt_qposadr[model.joint(n).id]
                              for n in ARM_JOINTS + ('r_joint',)])
        self.actuators = np.array([model.actuator(n+'_servo').id for n in ARM_JOINTS]
                                 + [model.actuator('gripper_servo').id])
        self.limits = model.actuator_ctrlrange[self.actuators].copy()
        self.rate = np.array([.8, .8, .8, .8, .8, 1.0])
        self.reference = data.qpos[self.qadr].copy()
        self.target = self.reference.copy()
        self.last_command = None
        self.timed_out = True
        self.write_controls()

    def state(self):
        q = self.data.qpos[self.qadr].copy()
        q[-1] = np.clip((q[-1]+.8)/1.25, 0, 1)
        return q

    def set_command(self, action, now):
        values = np.asarray(action, dtype=float)
        if (values.shape != (6,) or not np.isfinite(values).all()
                or not math.isfinite(now) or np.any(np.abs(values[:5]) > 2.09)
                or not 0 <= values[-1] <= 1):
            self.stop()
            return False
        self.target = values.copy()
        self.target[-1] = -.8 + 1.25 * values[-1]
        self.last_command = now
        self.timed_out = False
        return True

    def stop(self):
        # Keep the last bounded reference, including grip preload. Opening a
        # held object, or targeting measured finger position, would drop it.
        self.target[:] = self.reference
        self.last_command = None
        self.timed_out = True

    def write_controls(self):
        self.data.ctrl[self.actuators] = self.reference

    def apply(self, dt, now):
        if not math.isfinite(dt) or dt <= 0 or not math.isfinite(now):
            raise ValueError('step time must be finite; dt must be positive')
        if self.last_command is not None and now-self.last_command >= self.timeout:
            self.stop()
        self.reference += np.clip(self.target-self.reference, -self.rate*dt, self.rate*dt)
        self.reference = np.clip(self.reference, self.limits[:,0], self.limits[:,1])
        self.write_controls()
