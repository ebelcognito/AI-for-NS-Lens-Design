"""Physics, protocol, and accounting checks for the multi-body camera prototype."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from nslens.freeform_body.arc_polygon import ArcPolygonBody
from nslens.freeform_body.trace import InputRay, BoundaryEvent, ExitRay, TraceResult, trace_bundle
from nslens.camera.scene import OpticalScene
from nslens.camera.problem import CameraCodec, CameraBrief, evaluate
from nslens.camera.run import Session, inception


def slab(x, n=1.5):
    return ArcPolygonBody([(x,-2),(x+1,-2),(x+1,2),(x,2)],[0]*4,n)


class CameraTests(unittest.TestCase):
    def test_single_body_adapter_agrees(self):
        b=slab(1); rays=[InputRay((0,0),(1,.1))]
        a=trace_bundle(b,rays,max_interactions=16,energy_floor=1e-8)
        c=trace_bundle(OpticalScene([b]),rays,max_interactions=16,energy_floor=1e-8)
        self.assertAlmostEqual(a.escaped_energy,c.escaped_energy,places=12)
        self.assertEqual(len(a.exits),len(c.exits))
        for x,y in zip(a.exits,c.exits):
            np.testing.assert_allclose(x.direction,y.direction)
            self.assertEqual([e.kind for e in x.events],[e.kind for e in y.events])

    def test_two_material_slabs_and_revisits(self):
        scene=OpticalScene([slab(1,1.5),slab(4,1.7)])
        scene.validate()
        result=trace_bundle(scene,[InputRay((0,0),(1,0))],max_interactions=24,energy_floor=1e-9)
        dominant=result.dominant_exit()
        self.assertEqual([e.body_id for e in dominant.events],[0,0,1,1])
        expected=(1-((1.5-1)/(1.5+1))**2)**2*(1-((1.7-1)/(1.7+1))**2)**2
        self.assertAlmostEqual(dominant.energy,expected,places=12)
        self.assertLess(abs(result.closure_residual),1e-12)
        self.assertTrue(any([e.body_id for e in r.events].count(0)>2 for r in result.exits))

    def test_ideal_mirror_is_not_tir(self):
        scene=OpticalScene([slab(1)],[[False,False,False,True]])
        r=trace_bundle(scene,[InputRay((0,0),(1,0))])
        self.assertEqual(len(r.exits),1)
        self.assertEqual(r.exits[0].events[0].kind,'mirror')
        self.assertEqual(r.exits[0].events[0].face_id,3)
        self.assertEqual(r.exits[0].direction,(-1.,0.))
        self.assertEqual(r.exits[0].energy,1.)

    def test_branch_cap_conserves_energy(self):
        r=trace_bundle(OpticalScene([slab(1),slab(4)]),[InputRay((0,0),(1,0))],max_branches=2)
        self.assertGreater(r.branch_limited_energy,0)
        self.assertAlmostEqual(r.closure_residual,0.)

    def test_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError,'air gap'): OpticalScene([slab(1),slab(1.5)]).validate()

    def test_codec_roundtrip(self):
        for bound in (0.,.03):
            c=CameraCodec(departure_bound_mm=bound)
            g=np.random.default_rng(2).uniform(-1,1,c.gene_size)
            d=c.design(g); encoded=c.encode(d)
            a=c.decode(g); b=c.decode(encoded)
            for x,y in zip(a.bodies,b.bodies): np.testing.assert_allclose(x.corners,y.corners)
            self.assertEqual(a.mirrors,b.mirrors)
            bad=c.design(g); bad['bodies'][0]['center_mm'][0]=100
            with self.assertRaises(ValueError): c.encode(bad)

    def test_invalid_requests_cost_slots(self):
        with tempfile.TemporaryDirectory() as d:
            s=Session(Path(d),CameraCodec(),CameraBrief(),2,30)
            s.propose(design={}); s.propose(design={})
            self.assertFalse(s.available()); self.assertEqual(s.rays,0)
            with self.assertRaises(ValueError): s.propose(design={})
            self.assertFalse(s.finalize()['success'])

    def test_partial_population_charged_no_fictitious_update(self):
        with tempfile.TemporaryDirectory() as d:
            s=Session(Path(d),CameraCodec(),CameraBrief(),3,45)
            with patch('nslens.camera.run.CMAES.tell') as tell:
                inception(s,0); tell.assert_not_called()
            self.assertEqual(len(s.rows),3)

    def test_missing_light_never_passes_imaging(self):
        c=CameraCodec(corners=(3,)); d=c.design(np.zeros(c.gene_size))
        d['bodies'][0].update(center_mm=[18,0],half_size_mm=[1,1])
        row=evaluate(c,c.encode(d),CameraBrief())
        self.assertEqual(row['status'],'measured'); self.assertFalse(row['feasible'])
        self.assertEqual(row['quality'],0.)
        self.assertEqual(row['source_rays'],15)


    def test_imaging_metric_accepts_mapping_rejects_collapsed_fields(self):
        c=CameraCodec(corners=(3,)); d=c.design(np.zeros(c.gene_size))
        d['bodies'][0].update(center_mm=[18,0],half_size_mm=[1,1])
        g=c.encode(d); brief=CameraBrief()
        event=BoundaryEvent('refract',(18.,0.),0.,0.,0.,0.,None,1.,1.)
        def synthetic(collapse):
            rays=[]
            for i,field in enumerate(np.linspace(-11.6,11.6,3)):
                target=0. if collapse else 12.24*np.tan(np.deg2rad(field))
                for j in range(5):
                    rays.append(ExitRay(i*5+j,(23.,float(target)),(1.,0.),.5,.5,(event,)))
            return TraceResult(tuple(rays),(),15.,0.,0.,0.)
        with patch('nslens.camera.problem.trace_bundle',return_value=synthetic(False)):
            self.assertTrue(evaluate(c,g,brief)['feasible'])
        with patch('nslens.camera.problem.trace_bundle',return_value=synthetic(True)):
            row=evaluate(c,g,brief)
            self.assertFalse(row['feasible'])
            self.assertEqual(row['fields'][0]['rms_mm'],0.)
            self.assertGreater(row['violations']['centroid_0'],0.)

    def test_model_malformed_response_is_charged(self):
        from nslens.camera.run import model_run
        from unittest.mock import MagicMock
        response=MagicMock()
        response.__enter__.return_value.read.return_value=b'{"choices":[{"message":{"content":"broken"}}],"usage":{"total_tokens":12}}'
        with tempfile.TemporaryDirectory() as d:
            s=Session(Path(d),CameraCodec(),CameraBrief(),1,15)
            with patch('urllib.request.urlopen',return_value=response):
                result=model_run(s,'http://localhost/v1','fixture',0,100,1)
            self.assertEqual(result['candidates'],1)
            self.assertEqual(result['reported_total_tokens'],12)
            self.assertFalse(result['success'])


    def test_polynomial_multibody_trace_closes_energy(self):
        from nslens.freeform_body.polynomial_faces import PolynomialFaceBody
        bodies=[]
        for x in (1.,4.):
            base=slab(x)
            coefficients=np.zeros((4,3));coefficients[3,0]=.02
            bodies.append(PolynomialFaceBody(base.corners,base.sags,base.n,coefficients))
        scene=OpticalScene(bodies);scene.validate()
        result=trace_bundle(scene,[InputRay((0.,0.),(1.,.05))],energy_floor=1e-7)
        self.assertAlmostEqual(result.closure_residual,0.,places=12)
        self.assertEqual({e.body_id for e in result.dominant_exit().events},{0,1})


    def test_sequential_grade_advances_body_by_body_without_changing_acceptance(self):
        from nslens.camera.sequential_grade import grade
        from nslens.camera.run import camera_rank
        scene=OpticalScene([slab(1),slab(4),slab(7)])
        brief=CameraBrief()
        def event(body_id):
            return BoundaryEvent('refract',(body_id*3.+1.,0.),0.,0.,0.,0.,None,1.,1.,body_id,3)
        scores=[]
        for count in range(4):
            ray=ExitRay(0,(count*3.+.1,0.),(1.,0.),.5,.5,
                        tuple(event(i) for i in range(count)))
            trace=TraceResult((ray,),(),1.,0.,0.,0.)
            scores.append(grade(scene,trace,brief,1)['mean_score'])
        self.assertEqual(scores,sorted(scores))
        self.assertEqual(len(set(scores)),4)
        self.assertGreater(camera_rank(dict(status='measured',feasible=False,
            sequential_grade={'worst_score':scores[2],'mean_score':scores[2]},
            quality=0.,support=0.,violations={},progress=(0.,)))[2],scores[1])


    def test_sequential_error_prefers_nearer_next_surface(self):
        from nslens.camera.sequential_grade import grade
        scene=OpticalScene([slab(1),slab(4),slab(7)])
        event=BoundaryEvent('refract',(1.,0.),0.,0.,0.,0.,None,1.,1.,0,3)
        def score(origin,events):
            ray=ExitRay(0,origin,(1.,0.),.5,.5,events)
            trace=TraceResult((ray,),(),1.,0.,0.,0.)
            return grade(scene,trace,CameraBrief(),1)['mean_score']
        self.assertGreater(score((2.,0.),(event,)),score((2.,5.),(event,)))
        out_of_order=BoundaryEvent('refract',(7.,0.),0.,0.,0.,0.,None,1.,1.,2,3)
        self.assertLess(score((8.,0.),(out_of_order,)),score((2.,0.),(event,)))

if __name__=='__main__': unittest.main()
