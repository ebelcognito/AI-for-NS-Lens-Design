"""Insertion invariants independent of packaging feasibility."""
import json
from pathlib import Path
import numpy as np
import unittest
from nslens.spectrometer import powered_four_mirror as parent, powered_five_mirror as pm

def check_segment(segment):
    old=np.asarray(json.loads(Path('experiments/spectrometer_powered_four_mirror_cma_v3/result.json').read_text())['genes'])
    baseline=parent.trace(old,fields=3,waves=3,pupil_xy=np.array([[0.,0.],[.5,0],[-.5,0],[0,.5],[0,-.5]]))
    pm.configure(segment,100.)
    genes=pm.seed_genes(old,[.5,90.,180.])
    folded=pm.trace(genes,fields=3,waves=3,pupil_xy=np.array([[0.,0.],[.5,0],[-.5,0],[0,.5],[0,-.5]]))
    assert folded['model']['grating_index']==2+int(segment<=2)
    assert len(folded['model']['surfaces'])==6
    # Obstruction flags deliberately not asserted: these slots need not package.
    np.testing.assert_allclose(folded['xy'],baseline['xy'],atol=1e-10)
    np.testing.assert_allclose(folded['phase'],baseline['phase'],atol=1e-10)
    old_path=baseline['opl']+np.linalg.norm(baseline['image']-baseline['last_surface'],axis=-1)
    new_path=folded['opl']+np.linalg.norm(folded['image']-folded['last_surface'],axis=-1)
    np.testing.assert_allclose(new_path,old_path,atol=1e-10)

class FiveMirrorTests(unittest.TestCase):
    def test_all_six_insertion_slots(self):
        for segment in range(6):
            with self.subTest(segment=segment):check_segment(segment)

class HardwareObjectiveTests(unittest.TestCase):
    def test_opt_in_rejects_slit_crossing_but_keeps_feasible_geometry(self):
        data=json.loads(Path('experiments/spectrometer_powered_five_mirror_cma_v1/geometry_search.json').read_text())
        for evaluation,expected in [(2049,False),(1465,True)]:
            row=next(r for r in data['verified'] if r['evaluation']==evaluation)
            pm.configure(row['segment'],row['aperture'])
            ordinary=pm.evaluate(row['genes'],fields=3,waves=3,pupils=5)
            checked=pm.evaluate(row['genes'],fields=3,waves=3,pupils=5,hardware_constraints=True)
            self.assertNotIn('hardware',ordinary)
            self.assertEqual(checked['hardware']['passed'],expected)
            if expected:self.assertEqual(checked['loss'],ordinary['loss'])
            else:self.assertGreater(checked['loss'],ordinary['loss']+1.)
