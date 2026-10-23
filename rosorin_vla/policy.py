"""Load the fixed checkpoint locally; never use an expert or subsequent frames."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from .contracts import WEIGHTS_SHA, STATS_SHA

ROOT = Path(__file__).resolve().parents[1]

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

class LocalPolicy:
    def __init__(self, model_dir=None, device='cuda'):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

        self.torch = torch
        self.device = device
        folder = Path(model_dir or ROOT / 'models' / 'smolvla-block').resolve()
        for filename, expected in (('model.safetensors', WEIGHTS_SHA), ('stats.json', STATS_SHA)):
            path = folder / filename
            if not path.is_file():
                raise FileNotFoundError(f'{path} missing; run python -m rosorin_vla.download')
            if sha256(path) != expected:
                raise ValueError(f'CHECKPOINT_SHA256_MISMATCH: {filename}')
        self.contract = json.loads((folder / 'model_contract.json').read_text())
        for key, expected in {'observation_protocol':'initial_rgb_and_initial_state_only',
                'horizon':108, 'hz':10, 'initial_wait_s':0, 'postprocessing':'none', 'source':'mujoco_only'}.items():
            if self.contract.get(key) != expected:
                raise ValueError(f'MODEL_CONTRACT_MISMATCH: {key}')
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; install a compatible PyTorch build, or use --device cpu')
        torch.set_num_threads(4)
        backbone = folder / 'backbone'
        cfg = PreTrainedConfig.from_pretrained(str(folder), local_files_only=True)
        cfg.vlm_model_name = str(backbone)
        # The checkpoint already includes all retained VLM weights. Constructing
        # the architecture from config avoids an unnecessary 2GB base download.
        cfg.load_vlm_weights = False
        cfg.chunk_size = cfg.n_action_steps = 108
        cfg.n_obs_steps = 1
        cfg.num_steps = 10
        cfg.device = device
        cfg.push_to_hub = False
        self.policy = SmolVLAPolicy.from_pretrained(str(folder), config=cfg,
                                                  local_files_only=True, strict=True)
        self.policy.float().to(device).eval()
        self.policy.config.num_steps = self.policy.model.config.num_steps = 10
        stats = {k:{n:torch.tensor(v) for n,v in values.items()}
                 for k,values in json.loads((folder / 'stats.json').read_text()).items()}
        self.pre, self.post = make_smolvla_pre_post_processors(cfg, stats)

    def predict(self, observation, request_id):
        torch = self.torch
        rgb, state = observation['rgb'], observation['state']
        if rgb.shape != (400,640,3) or rgb.dtype != np.uint8:
            raise ValueError('INITIAL_RGB_MUST_BE_UINT8_400_640_3')
        if state.shape != (6,) or state.dtype != np.float32 or not np.isfinite(state).all():
            raise ValueError('INITIAL_STATE_MUST_BE_FLOAT32_6')
        fingerprint = hashlib.sha256(rgb.tobytes()+state.tobytes()).hexdigest()
        started = time.monotonic()
        with torch.inference_mode():
            batch = {'observation.images.camera1':torch.tensor(rgb).permute(2,0,1).float().unsqueeze(0)/255,
                     'observation.state':torch.tensor(state).unsqueeze(0), 'task':[self.contract['task']]}
            noise = torch.randn((1,108,self.policy.config.max_action_dim), device=self.device,
                                generator=torch.Generator(device=self.device).manual_seed(731))
            self.policy.reset()
            actions = self.post(self.policy.predict_action_chunk(self.pre(batch), noise=noise)).float().cpu().numpy()[0]
        if actions.shape != (108,6) or not np.isfinite(actions).all():
            raise ValueError('INVALID_RAW_MODEL_OUTPUT')
        return {'request_id':request_id, 'initial_sha256':fingerprint, 'actions':actions.tolist(),
                'policy_calls':1, 'elapsed_s':time.monotonic()-started,
                'weights_sha256':WEIGHTS_SHA, 'stats_sha256':STATS_SHA,
                'precision':'fp32', 'native_denoising_steps':10, 'noise_seed':731,
                'postprocessing':'none', 'horizon':108, 'hz':10, 'initial_wait_s':0,
                'input_fields':['rgb','state'], 'device':self.device,
                'backend':'local_checkpoint'}
