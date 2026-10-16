import copy
import unittest
import numpy as np
from nslens.spectrometer import fabrication_probe as p
from nslens.spectrometer import powered_five_mirror as five

class FabricationProbeTests(unittest.TestCase):
    def test_zero_width_matches_existing_five_mirror_trace(self):
        import json
        data=json.loads(p.CANDIDATE.read_text());layout=p.load_layout('five_mirror')
        five.configure(layout['insertion_segment'],layout['surfaces'][layout['insertion_segment']]['aperture'])
        old=five.trace(data['genes'],fields=3,waves=3,pupils=5)
        xy,good=p.trace_layout(layout,p.rays(layout,3,3,5,1,0))
        np.testing.assert_array_equal(good,old['good'])
        np.testing.assert_allclose(xy[good],old['xy'][good],atol=1e-10,rtol=0)
    def test_component_motion_does_not_retarget_other_components_or_launch(self):
        m=p.load_layout('five_mirror');original=copy.deepcopy(m)
        moved=p.perturb(m,0,0,10,30)
        np.testing.assert_array_equal(m['surfaces'][0]['vertex'],original['surfaces'][0]['vertex'])
        np.testing.assert_allclose(moved['surfaces'][0]['vertex']-m['surfaces'][0]['vertex'],[.01,0,0],atol=1e-14)
        for i in range(1,len(m['surfaces'])):np.testing.assert_array_equal(moved['surfaces'][i]['vertex'],m['surfaces'][i]['vertex'])
        np.testing.assert_array_equal(p.rays(m,3,3,5,3,25)[1],p.rays(moved,3,3,5,3,25)[1])
        self.assertAlmostEqual(np.linalg.det(moved['surfaces'][0]['basis']),np.linalg.det(m['surfaces'][0]['basis']),places=12)
    def test_finite_slit_and_pixels_add_width(self):
        m=p.load_layout('five_mirror')
        a=p.metrics(m,p.rays(m,3,3,7,1,0),pixel_um=0)
        b=p.metrics(m,p.rays(m,3,3,7,7,25),pixel_um=10)
        self.assertGreater(b['max_spectral_rms_nm_with_pixel_variance'],a['max_spectral_rms_nm_with_pixel_variance'])
        self.assertEqual(b['launched_rays'],a['launched_rays']*7)
    def test_no_rays_not_reported_as_zero_blur(self):
        m=p.load_layout('five_mirror');launch=p.rays(m,3,3,5,3,25)
        m['surfaces'][0]['vertex']+=np.array([1000.,0,0])
        row=p.metrics(m,launch)
        self.assertFalse(row['all_cells_have_rays']);self.assertIsNone(row['max_spectral_rms_nm_with_pixel_variance'])
    def test_focus_grid_includes_uncompensated_case(self):
        m=p.load_layout('five_mirror');launch=p.rays(m,2,3,3,3,25)
        row=p.metrics(m,launch);ref=np.asarray(row['centroids_mm'])
        best=p.compensate(m,launch,10,ref,20)
        self.assertLessEqual(p.score(best['metrics']),p.score(row)+1e-12)
        self.assertLessEqual(abs(best['detector_focus_um']),20)

if __name__=='__main__':unittest.main()
