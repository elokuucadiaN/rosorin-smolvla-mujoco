"""Standalone MuJoCo viewer + real VLA inference; ROS is not required."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',type=Path)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--endpoint',help='Optional loopback inference server, e.g. http://127.0.0.1:8767')
    parser.add_argument('--seed',type=int,default=0,help='Six fixed scene positions, selected by seed %% 6')
    parser.add_argument('--episodes',type=int,default=1,help='Sequential positions; a fresh prediction for each')
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('outputs'))
    parser.add_argument('--inference-timeout',type=float,default=600.)
    args=parser.parse_args()
    if args.episodes<1 or not 0<args.inference_timeout<float('inf'):
        parser.error('episodes and inference-timeout must be positive')
    if args.headless:
        os.environ.setdefault('MUJOCO_GL','egl' if Path('/dev/nvidia0').exists() else 'osmesa')
    import mujoco
    import numpy as np
    from rosorin_mujoco_sim.factory_manipulation_env import ManipulationEnv
    from rosorin_mujoco_sim.vla_execution import VLAClient,VLAExecutor
    from .policy import LocalPolicy
    args.output.mkdir(parents=True,exist_ok=True)
    env=ManipulationEnv()
    def forbidden(*a,**kw):raise RuntimeError('EXPERT_DISABLED_IN_VLA_MODE')
    env.expert_plan=env.solve=forbidden
    print('Loading fixed SmolVLA checkpoint; simulation has not stepped.',flush=True)
    client=VLAClient(args.endpoint,timeout=args.inference_timeout) if args.endpoint else LocalPolicy(args.model_dir,args.device)
    viewer=None
    if not args.headless:
        import mujoco.viewer
        viewer=mujoco.viewer.launch_passive(env.model,env.data)
        with viewer.lock():
            viewer.cam.lookat[:]=[.19,.025,.12]
            viewer.cam.distance=.42
            viewer.cam.azimuth=135
            viewer.cam.elevation=-20
            viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
            viewer.opt.geomgroup[3]=0
    reports=[]
    try:
        for episode in range(args.episodes):
            env.reset(args.seed+episode)
            executor=VLAExecutor(env,client)
            executor.begin()
            deadline=time.monotonic()+args.inference_timeout
            while executor.state=='inferring':
                executor.poll()
                if viewer:
                    if not viewer.is_running():executor.cancel();return 130
                    viewer.sync(state_only=True)
                if time.monotonic()>deadline:
                    executor.cancel();raise TimeoutError('Inference timed out before motion')
                time.sleep(.01)
            if executor.state!='executing':
                raise RuntimeError('VLA inference rejected: '+executor.error)
            raw=executor.actions.copy()
            print(f'Executing seed {env.seed}: 108 raw commands, 10Hz, 10.8s; one initial image.',flush=True)
            start=time.monotonic()
            while executor.state=='executing':
                executor.step(time.monotonic())
                if viewer and executor.steps%40==0:
                    if not viewer.is_running():executor.cancel();break
                    viewer.sync(state_only=True)
                    time.sleep(max(0.,float(env.data.time)-(time.monotonic()-start)))
            result=executor.result()
            np.testing.assert_array_equal(raw,executor.actions) if executor.actions is not None else None
            stem=f'episode_{episode:03d}_seed_{env.seed}'
            np.savez_compressed(args.output/(stem+'.npz'),actions=raw.astype(np.float32),
                                initial_rgb=executor.observation['rgb'],initial_state=executor.observation['state'])
            (args.output/(stem+'.json')).write_text(json.dumps(result,indent=2)+'\n')
            reports.append(result)
            print(json.dumps({'seed':env.seed,'success':result['success'],'state':executor.state,
                              'policy_calls':result['policy_calls'],'camera_reads':result['policy_camera_reads']}),flush=True)
            if executor.state=='cancelled':return 130
        (args.output/'summary.json').write_text(json.dumps({'episodes':len(reports),
                    'passed':sum(r['success'] for r in reports),'results':reports},indent=2)+'\n')
        if viewer:
            print('Episode complete. Close the window or press Ctrl+C to exit; physics is paused.',flush=True)
            while viewer.is_running():viewer.sync(state_only=True);time.sleep(.03)
        return 0 if all(r['success'] for r in reports) else 1
    except KeyboardInterrupt:
        if 'executor' in locals():executor.cancel()
        return 130
    finally:
        if viewer:viewer.close();time.sleep(.3)

if __name__=='__main__':
    try:sys.exit(main() or 0)
    except (ValueError,RuntimeError,FileNotFoundError,TimeoutError) as exc:
        print(str(exc),file=sys.stderr);sys.exit(2)
