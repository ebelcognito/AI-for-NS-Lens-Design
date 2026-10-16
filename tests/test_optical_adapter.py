import tempfile
import unittest
from pathlib import Path
import numpy as np
try:
    import torch
    import transformers
except ImportError:
    torch = None
if torch is not None:
    from nslens.optical_adapter.network import tiny_qwen, NumericPolicy, save_adapter, load_adapter
from nslens.optical_adapter.state import observe, apply_action
from nslens.optical_adapter.data import validate_dataset


class OpticalAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch is not None: torch.set_num_threads(2)

    @unittest.skipIf(torch is None, "optional torch/transformers runtime not installed")
    def test_real_qwen_gradient_flow_and_frozen_backbone(self):
        policy=tiny_qwen();policy.train()
        frozen={k:v.clone() for k,v in policy.backbone.state_dict().items()}
        cells=torch.randn(2,9,10);state=torch.randn(2,37)
        params=[p for p in policy.parameters() if p.requires_grad]
        opt=torch.optim.AdamW(params,lr=.001)
        action=policy(cells,state)
        self.assertEqual(action.shape,(2,36));self.assertTrue((abs(action)<1).all())
        (action-.3).square().mean().backward()
        for module in (policy.encoder.cell,policy.encoder.global_state,policy.cell_projection,policy.head):
            self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None),0)
        self.assertTrue(all(p.grad is None for p in policy.backbone.parameters()))
        opt.step()
        for k,v in policy.backbone.state_dict().items():torch.testing.assert_close(v,frozen[k],rtol=0,atol=0)
        policy.eval()
        with torch.no_grad():
            a=policy(cells,state);b=policy(cells+1,state)
        self.assertGreater((a-b).abs().max().item(),1e-6)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'adapter.pt';save_adapter(policy,path)
            saved=torch.load(path,weights_only=True)
            self.assertFalse(any(k.startswith('backbone.') for k in saved['weights']))
            restored=load_adapter(tiny_qwen(),path).eval()
            with torch.no_grad():torch.testing.assert_close(restored(cells,state),a,rtol=0,atol=0)
            saved['simulator_sha256']='different-simulator'
            torch.save(saved,path)
            with self.assertRaises(ValueError):load_adapter(tiny_qwen(),path)

    def test_observation_missing_rays_and_bounded_actions(self):
        g=np.zeros(36);cells,state=observe(g)
        self.assertEqual(cells.shape,(9,10));self.assertEqual(state.shape,(37,))
        self.assertTrue(np.isfinite(cells).all())
        self.assertTrue(((cells[:,2]>=0)&(cells[:,2]<=1)).all())
        self.assertTrue(np.array_equal(cells[:,8],(cells[:,2]==0).astype(float)))
        extreme=np.ones(36)*.99;candidate=apply_action(extreme,np.ones(36),.05)
        self.assertTrue((candidate<=1).all());self.assertLessEqual(abs(candidate-extreme).max(),.05)
        for a in (np.full(36,np.nan),np.ones(36)*2,np.zeros(35)):
            with self.assertRaises(ValueError):apply_action(g,a,.05)

    @unittest.skipIf(torch is None, "optional torch/transformers runtime not installed")
    def test_numeric_encoder_is_trainable(self):
        policy=NumericPolicy();out=policy(torch.randn(2,9,10),torch.randn(2,37))
        out.square().mean().backward()
        self.assertTrue(all(p.grad is not None for p in policy.parameters()))

    def test_ancestry_leakage_rejected(self):
        from nslens.optical_adapter.state import SCHEMA
        with self.assertRaises(ValueError):
            validate_dataset(dict(schema=SCHEMA,rows=[dict(family='same',split='train'),dict(family='same',split='validation')]))


if __name__=='__main__':unittest.main()
