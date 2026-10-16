"""Independent regression gates for an explicitly reserved slit assembly volume."""
import json
from pathlib import Path
import unittest
import numpy as np
from nslens.spectrometer.slit_hardware import hardware_check, segments_intersect_box, RAY_MARGIN_MM


def scalar_segment_box(a,b,half=(6.,7.,1.)):
    """Independent scalar slab oracle, including tangent and endpoint contact."""
    lower,upper=0.,1.
    for x,y,h in zip(a,b,half):
        direction=y-x
        if direction==0:
            if x < -h or x > h:return False
        else:
            p,q=(-h-x)/direction,(h-x)/direction
            lower=max(lower,min(p,q));upper=min(upper,max(p,q))
            if lower>upper:return False
    return True


def clean_layout():
    return dict(origin=np.zeros(3),surfaces=[dict(vertex=np.array([100.,0.,10.]),basis=np.eye(3),coeff=np.zeros(9),aperture=2.)],input_length=60.,image=np.array([100.,0.,20.]),image_x=np.array([1.,0.,0.]),image_y=np.array([0.,1.,0.]),image_normal=np.array([0.,0.,1.]),detector_bounds=np.array([[-2.,-2.],[2.,2.]]))


def paths(a,b):
    return np.array([[[0.,0.,0.]], [a], [b]])


class SlitHardwareTests(unittest.TestCase):
    def test_initial_launch_segment_is_exempt_but_return_segment_is_not(self):
        self.assertTrue(hardware_check(clean_layout(),paths=paths([0,0,10],[0,0,20]))['passed'])
        self.assertFalse(hardware_check(clean_layout(),paths=paths([0,0,10],[0,0,-10]))['passed'])

    def test_continuous_segment_tests_agree_with_independent_slab_oracle(self):
        cases=[([-10,0,0],[10,0,0]),([-10,7,1],[10,7,1]),([6,7,1],[10,10,10]),([0,0,0],[0,0,0]),([7,0,-2],[7,0,2]),([-10,7.001,0],[10,7.001,0]),([0,0,2],[0,0,20])]
        for a,b in cases:
            with self.subTest(a=a,b=b):
                self.assertEqual(bool(segments_intersect_box(a,b)),scalar_segment_box(a,b))
                expanded=tuple(np.array([6.,7.,1.])+RAY_MARGIN_MM)
                self.assertEqual(hardware_check(clean_layout(),paths=paths(a,b))['passed'],not scalar_segment_box(a,b,expanded))

    def test_nonfinite_segments_are_conservatively_blocked(self):
        for bad in [np.nan,np.inf,-np.inf]:
            q=paths([0,0,10],[0,0,20]);q[2,0,0]=bad
            self.assertTrue(bool(segments_intersect_box(q[1,0],q[2,0])))
            self.assertFalse(hardware_check(clean_layout(),paths=q)['passed'])

    def test_optical_disk_crossing_box_rejected_even_if_center_outside(self):
        m=clean_layout();m['surfaces'][0]['vertex']=np.array([7.,0.,0.]);m['surfaces'][0]['aperture']=2.
        self.assertFalse(hardware_check(m,paths=paths([0,0,10],[0,0,20]))['passed'])

    def test_detector_rectangle_crossing_box_rejected(self):
        m=clean_layout();m['image']=np.array([0.,0.,0.]);m['detector_bounds']=np.array([[-20.,-20.],[20.,20.]])
        self.assertFalse(hardware_check(m,paths=paths([0,0,10],[0,0,20]))['passed'])

    def test_upstream_optical_obstruction_rejected_outside_box(self):
        m=clean_layout();m['surfaces'][0]['vertex']=np.array([0.,0.,-10.]);m['surfaces'][0]['aperture']=30.
        self.assertFalse(hardware_check(m,paths=paths([0,0,10],[0,0,20]))['passed'])

    def test_old_five_mirror_candidate_fails_reserved_slit_volume(self):
        from nslens.spectrometer import powered_five_mirror as p
        root=Path(__file__).resolve().parents[1]
        r=json.loads((root/'experiments/spectrometer_powered_five_mirror_cma_v1/result.json').read_text())
        previous=(p.SEGMENT,p.APERTURE)
        try:
            p.configure(r['configuration']['segment'],r['configuration']['aperture']);t=p.trace(r['genes'],fields=3,waves=3,pupils=3)
            self.assertTrue(any(scalar_segment_box(a,b) for j in range(1,len(t['paths'])-1) for a,b in zip(t['paths'][j],t['paths'][j+1])))
            self.assertFalse(hardware_check(t['model'],paths=t['paths'])['passed'])
        finally:p.configure(*previous)

    def test_old_four_mirror_candidate_fails_hardware_gate(self):
        from nslens.spectrometer import powered_four_mirror as p
        root=Path(__file__).resolve().parents[1];r=json.loads((root/'experiments/spectrometer_powered_four_mirror_cma_v3/result.json').read_text());t=p.trace(r['genes'],fields=3,waves=3,pupils=3)
        self.assertFalse(hardware_check(t['model'],paths=t['paths'])['passed'])

if __name__=='__main__':unittest.main()
