"""Exact meridional sag adapters for the repository's Fresnel/TIR tracer.

Coordinates are (patent Z, patent Y), in mm. Polynomial coefficients are never
refitted. The finite edge cuts are inferred separately from optical prescriptions.
"""
from pathlib import Path
from types import SimpleNamespace
from math import factorial
import json
import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import root, brentq
from matplotlib.path import Path as PolygonPath


def load_prescription():
    return json.loads(Path(__file__).with_name('prescription.json').read_text())


def codev_mode(j):
    """One-based Born-Wolf ordering: n ascending, cos(n..1), m=0, sin(1..n)."""
    if j < 1:
        raise ValueError('Zernike index must be positive')
    n = 0
    while (n+1)*(n+2)//2 < j:
        n += 1
    k = j - n*(n+1)//2 - 1
    modes = [(m, 'cos') for m in range(n, 0, -2)]
    if n % 2 == 0:
        modes.append((0, 'cos'))
    modes.extend((m, 'sin') for m in range(1 if n % 2 else 2, n+1, 2))
    m, trig = modes[k]
    return n, m, trig


def meridional_zernike(j, radius):
    """Exact polynomial at local X=0, with signed Y; no RMS normalization."""
    n, m, trig = codev_mode(j)
    phase = np.cos(m*np.pi/2) if trig == 'cos' else np.sin(m*np.pi/2)
    coeff = np.zeros(n+1)
    if abs(phase) < 1e-12:
        return Polynomial([0.])
    for k in range((n-m)//2+1):
        power = n-2*k
        coeff[power] = phase*(-1)**k*factorial(n-k)/(
            factorial(k)*factorial((n+m)//2-k)*factorial((n-m)//2-k)*radius**power)
    return Polynomial(coeff)


class Surface:
    def __init__(self, spec, body_id, n):
        self.spec = spec
        self.name = spec['name']
        self.body_id, self.n = body_id, n
        self.mirror = self.name in ('P1S2', 'P2S3')
        self.origin = np.array(spec['origin_zy_mm'])
        a = np.deg2rad(spec['tilt_deg'])
        self.tangent = np.array([np.sin(a), np.cos(a)])
        self.axis = np.array([np.cos(a), -np.sin(a)])
        self.radius = spec.get('radius_mm', None)
        self.poly = Polynomial([0.])
        for term, value in spec['coefficients'].items():
            if spec['kind'] == 'zernike_codev':
                self.poly += value*meridional_zernike(int(term[2:]), spec['normalization_radius_mm'])
            elif spec['kind'] == 'asphere':
                self.poly += Polynomial([0.]*int(term)+[value])
            elif term.startswith('Y') and 'X' not in term:
                self.poly += Polynomial([0.]*int(term[1:] or '1')+[value])
        self.dp = self.poly.deriv()
        self.ddp = self.dp.deriv()
        self.limits = (-3.,3.)

    def sag(self, u, derivative=0):
        u = np.asarray(u)
        value = (self.poly, self.dp, self.ddp)[derivative](u)
        if self.radius is not None:
            r = self.radius
            with np.errstate(invalid='ignore', divide='ignore'):
                q = np.sqrt(1-(u/r)**2)
                value = value + (u*u/(r*(1+q)) if derivative == 0 else
                                 u/(r*q) if derivative == 1 else 1/(r*q**3))
        return value

    def point(self, u):
        u = np.asarray(u)
        return self.origin + u[...,None]*self.tangent + self.sag(u)[...,None]*self.axis

    def prepare(self, count):
        self.grid = np.linspace(*self.limits, count+1)
        self.points = self.point(self.grid)

    def intersect(self, o, d, t_min):
        # Cross(point(u)-o, d)=0, independently bracketed on every face.
        p = self.points-o
        values = p[:,0]*d[1]-p[:,1]*d[0]
        indices = np.flatnonzero(values[:-1]*values[1:] <= 0)
        best = None
        for k in indices:
            def f(u):
                p = self.point(u)-o
                return p[0]*d[1]-p[1]*d[0]
            u = brentq(f, self.grid[k], self.grid[k+1], xtol=1e-13)
            point = self.point(u)
            t = float(np.dot(point-o,d))
            if t <= t_min or (best is not None and t >= best.t):
                continue
            tangent = self.tangent+self.sag(u,1)*self.axis
            normal = np.array([tangent[1],-tangent[0]])
            normal /= np.linalg.norm(normal)
            curvature = float(self.sag(u,2)/(1+self.sag(u,1)**2)**1.5)
            best = SimpleNamespace(t=t, point=tuple(point), outward_normal=tuple(normal),
                u=float(u), arc_s=0., curvature=curvature, body_id=self.body_id,
                face_id=self.face_id, material_index=self.n, ideal_mirror=self.mirror)
        return best


def corner(a, b, guess):
    def fun(uv): return a.point(uv[0])-b.point(uv[1])
    sol = root(fun, guess, tol=1e-11)
    if not sol.success or np.linalg.norm(fun(sol.x))>1e-8:
        raise ValueError(f'Cannot close {a.name}/{b.name}: {sol.message}')
    return sol.x


class PatentScene:
    """Nearest intersection adapter; event ordering is determined by geometry."""
    def __init__(self, prescription=None, scan=256):
        self.prescription = prescription or load_prescription()
        self.surfaces = []
        for spec in self.prescription['surfaces']:
            body = ['L1','P1','P2'].index(spec['name'][:2])
            self.surfaces.append(Surface(spec,body,self.prescription['materials'][spec['name'][:2]]['nd']))
        self.n = self.surfaces[0].n
        self.polygons=[]
        for s in self.surfaces[:2]: s.limits=(-3.15,3.15)
        # Solve the three optical faces' intersections. These are mechanical edge
        # assumptions, not additional optical coefficients or constrained ray routes.
        for start in (2,5):
            a,b,c = self.surfaces[start:start+3]
            guesses = [(5.,3.), (-5.,0.), (-3.,0.)] if start==2 else [(0.,4.),(-4.,0.),(0.,-2.)]
            ab = corner(a,b,guesses[0]); bc=corner(b,c,guesses[1]); ca=corner(c,a,guesses[2])
            a.limits=tuple(sorted([ab[0],ca[1]]))
            b.limits=tuple(sorted([ab[1],bc[0]]))
            c.limits=tuple(sorted([bc[1],ca[0]]))
            chain=np.concatenate([a.point(np.linspace(ca[1],ab[0],200)),b.point(np.linspace(ab[1],bc[0],200)),c.point(np.linspace(bc[1],ca[0],200))])
            self.polygons.append(chain)
        lens=np.concatenate([self.surfaces[0].point(np.linspace(-3.15,3.15,200)), self.surfaces[1].point(np.linspace(3.15,-3.15,200))])
        self.polygons.insert(0,lens)
        for i,s in enumerate(self.surfaces):
            s.face_id=i
            s.prepare(scan)
        # Lens side walls close the solid physically; represented as straight sag faces.
        for y in (-3.15,3.15):
            p,q=(s.point(y) for s in self.surfaces[:2])
            v=q-p;length=np.linalg.norm(v)
            spec=dict(name='L1edge',origin_zy_mm=((p+q)/2).tolist(),tilt_deg=float(np.rad2deg(np.arctan2(v[0],v[1]))),kind='xy',coefficients={})
            s=Surface(spec,0,self.n);s.limits=(-length/2,length/2);s.face_id=len(self.surfaces);s.prepare(1);self.surfaces.append(s)
        self.paths=[PolygonPath(p) for p in self.polygons]

    def validate(self):
        """Reject overlapping solids: the reused tracer requires disjoint media."""
        for i, a in enumerate(self.paths):
            for j, b in enumerate(self.paths[:i]):
                if a.intersects_path(b, filled=True):
                    raise ValueError(f'Optical bodies {j} and {i} overlap')
        for surface in self.surfaces:
            if not np.isfinite(surface.points).all():
                raise ValueError(f'Nonfinite surface: {surface.name}')

    def contains(self, point):
        return any(p.contains_point(point) for p in self.paths)

    def nearest_intersection(self, origin, direction, t_min=1e-8, **kwargs):
        hits=[s.intersect(np.asarray(origin),np.asarray(direction),t_min) for s in self.surfaces]
        return min((h for h in hits if h is not None),key=lambda h:h.t,default=None)
