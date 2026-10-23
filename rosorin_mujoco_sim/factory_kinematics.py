"""Factory FK and mimic helpers vendored from the accepted reference."""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

def transform(joint):
    origin=joint.find('origin')
    pose=np.eye(4)
    if origin is not None:
        pose[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
        pose[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
    return pose

def fk(joints,frame,angles):
    if frame=='base_footprint':
        return np.eye(4)
    joint=joints[frame]
    pose=fk(joints,joint.find('parent').get('link'),angles)@transform(joint)
    if joint.get('type') in ('revolute','continuous'):
        axis=np.fromstring(joint.find('axis').get('xyz'),sep=' ')
        motion=np.eye(4)
        motion[:3,:3]=Rotation.from_rotvec(axis/np.linalg.norm(axis)*angles.get(joint.get('name'),0.)).as_matrix()
        pose=pose@motion
    return pose

def set_state(model,data,joints,angles):
    full=dict(angles)
    # Resolve the vendor mimic graph rather than assuming five fixed signs.
    remaining={j.get('name'):j.find('mimic') for j in joints.values() if j.find('mimic') is not None}
    while remaining:
        progress=False
        for name,mimic in list(remaining.items()):
            parent=mimic.get('joint')
            if parent in full:
                full[name]=float(mimic.get('multiplier','1'))*full[parent]+float(mimic.get('offset','0'))
                del remaining[name];progress=True
        if not progress:
            raise ValueError('MIMIC_GRAPH_UNRESOLVED')
    for name,angle in full.items():
        joint=model.joint(name)
        if not model.jnt_range[joint.id,0]-1e-9<=angle<=model.jnt_range[joint.id,1]+1e-9:
            raise ValueError(f'OUTSIDE_FACTORY_JOINT_LIMIT: {name}={angle}')
        data.qpos[model.jnt_qposadr[joint.id]]=angle
    mujoco.mj_forward(model,data)
    return full

def mesh_vertices_world(model,data,name):
    mesh=model.mesh(name).id
    geom=np.flatnonzero((model.geom_type==mujoco.mjtGeom.mjGEOM_MESH)&(model.geom_dataid==mesh))[0]
    start=model.mesh_vertadr[mesh];count=model.mesh_vertnum[mesh]
    vertices=model.mesh_vert[start:start+count]
    return vertices@data.geom_xmat[geom].reshape(3,3).T+data.geom_xpos[geom]
