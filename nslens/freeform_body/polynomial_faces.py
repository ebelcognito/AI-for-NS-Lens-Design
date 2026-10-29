"""Small, endpoint-clamped polynomial departures on individual circular faces.

Each face keeps its circular base and receives a chord-normal displacement
16*f**2*(1-f)**2 * (c0 + c1*(2*f-1) + c2*(6*f*f-6*f+1)), in mm.
The displacement and its first derivative vanish at both corners. Intersections
are bracketed independently on each face and refined on the continuous surface;
scan-density convergence must be checked for final optical claims.
"""
from dataclasses import replace
from math import cos, sin, hypot
import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import brentq
from .arc_polygon import ArcPolygonBody
from .boundary import BoundaryHit


class PolynomialFaceBody(ArcPolygonBody):
    def __init__(self, corners, sags, n, coefficients, *, face_scan=64,
                 force_numerical=False):
        # Coefficients refer to the canonical CCW face ordering. Require that order.
        corners = np.asarray(corners, dtype=float)
        cross = np.sum(corners[:, 0]*np.roll(corners[:, 1], -1)
                       - corners[:, 1]*np.roll(corners[:, 0], -1))
        if cross <= 0:
            raise ValueError('polynomial faces require counter-clockwise corners')
        super().__init__(corners, sags, n)
        c = np.asarray(coefficients, dtype=float)
        if c.shape != (self.control_count, 3) or not np.isfinite(c).all():
            raise ValueError('coefficients must have shape (faces, 3) and be finite')
        if face_scan < 16:
            raise ValueError('face_scan must be at least 16')
        self.coefficients = c.copy()
        self.face_scan = int(face_scan)
        self._zero = not np.any(c) and not force_numerical
        chord = np.roll(self._corners, -1, axis=0)-self._corners
        self._chord = chord
        self._chord_normals = np.column_stack([chord[:, 1], -chord[:, 0]]) / np.linalg.norm(chord, axis=1)[:, None]
        f = Polynomial([0, 1]); envelope = 16*f*f*(1-f)*(1-f)
        self._polys = [envelope*(v[0]+v[1]*(2*f-1)+v[2]*(6*f*f-6*f+1)) for v in c]
        self._derivs = [[p.deriv(k) for p in self._polys] for k in range(3)]
        self._scalar_coeffs = [[tuple(p.coef) for p in ps] for ps in self._derivs]
        self._scan_f = np.linspace(0, 1, self.face_scan+1)
        self._scan_points = np.array([self._face_points(i, self._scan_f) for i in range(self.control_count)])
        us = np.unique(np.r_[np.linspace(0, 1, 4097), self._cumulative])
        points = self.evaluate(us % 1)
        lengths = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        self._arc_u = us
        self._physical_perimeter = float(lengths[-1])
        self._arc_s = lengths / lengths[-1]

    @property
    def perimeter(self):
        return self._physical_perimeter

    @staticmethod
    def _horner(c, x):
        value = 0.
        for a in reversed(c): value = value*x+a
        return value

    def _local(self, index, f, derivative=0):
        e = self._edges[index]
        if e['kind'] == 'line':
            if derivative == 0: x,y = self._corners[index] + f*self._chord[index]
            elif derivative == 1: x,y = self._chord[index]
            else: x,y = 0.,0.
        else:
            theta = e['theta0']+f*e['sweep']; radius=e['radius']; w=e['sweep']
            if derivative == 0: x,y=e['center'][0]+radius*cos(theta),e['center'][1]+radius*sin(theta)
            elif derivative == 1: x,y=-radius*w*sin(theta),radius*w*cos(theta)
            else: x,y=-radius*w*w*cos(theta),-radius*w*w*sin(theta)
        delta=self._horner(self._scalar_coeffs[derivative][index],f)
        normal=self._chord_normals[index]
        return x+delta*normal[0],y+delta*normal[1]

    def _face_points(self,index,f):
        e=self._edges[index];f=np.asarray(f)
        if e['kind']=='line':p=self._corners[index]+f[...,None]*self._chord[index]
        else:
            theta=e['theta0']+f*e['sweep']
            p=e['center']+e['radius']*np.stack([np.cos(theta),np.sin(theta)],axis=-1)
        return p+self._polys[index](f)[...,None]*self._chord_normals[index]

    def evaluate(self,u):
        scalar=np.ndim(u)==0;u=np.atleast_1d(u);index,f=self._locate_many(u)
        result=np.empty((len(u),2))
        for i in range(self.control_count):
            mask=index==i;result[mask]=self._face_points(i,f[mask])
        return result[0] if scalar else result

    def _point_on(self,index,fraction):
        return np.asarray(self._local(index,fraction))

    def _normal_on(self,index,fraction):
        x,y=self._local(index,fraction,1);speed=hypot(x,y)
        return np.asarray([y/speed,-x/speed])

    def outward_normal(self,u):
        scalar=np.ndim(u)==0;index,f=self._locate_many(np.atleast_1d(u))
        value=np.array([self._normal_on(int(i),float(t)) for i,t in zip(index,f)])
        return value[0] if scalar else value

    def curvature(self,u):
        scalar=np.ndim(u)==0;index,f=self._locate_many(np.atleast_1d(u));out=[]
        for i,t in zip(index,f):
            x,y=self._local(i,t,1);xx,yy=self._local(i,t,2)
            out.append((x*yy-y*xx)/hypot(x,y)**3)
        return out[0] if scalar else np.asarray(out)

    def arclength_coordinates(self,u,table_size=2048):
        return np.interp(np.asarray(u)%1,self._arc_u,self._arc_s)

    def geometry_report(self,sample_count=256):
        r=super().geometry_report(sample_count)
        u=np.linspace(0,1,max(256,sample_count),endpoint=False)
        return replace(r,perimeter=self.perimeter,max_abs_curvature=float(np.max(abs(self.curvature(u)))))

    def extent(self,scan=128):
        points=list(self._corners)
        for i in range(self.control_count):
            f=np.linspace(0,1,scan+1)
            for axis in (0,1):
                deriv=np.array([self._local(i,t,1)[axis] for t in f])
                for j in np.flatnonzero(deriv[:-1]*deriv[1:]<0):
                    root=brentq(lambda t:self._local(i,t,1)[axis],f[j],f[j+1],xtol=1e-13)
                    points.append(self._local(i,root))
        points=np.asarray(points)
        return points.min(axis=0),points.max(axis=0)

    def all_intersections(self,origin,direction,*,t_min=1e-7,scan_count=0):
        if self._zero:
            return super().all_intersections(origin,direction,t_min=t_min)
        o=np.asarray(origin,dtype=float);d=np.asarray(direction,dtype=float)
        if o.shape!=(2,) or d.shape!=(2,) or not np.isfinite(o).all() or not np.isfinite(d).all() or np.linalg.norm(d)==0:
            raise ValueError('finite origin and nonzero direction required')
        d=d/np.linalg.norm(d);ox,oy=o;dx,dy=d
        rel=self._scan_points-o
        residual=rel[:,:,0]*dy-rel[:,:,1]*dx
        hits=[]
        for i in range(self.control_count):
            values=residual[i]
            candidates=np.flatnonzero((values[:-1]*values[1:]<0)|(abs(values[:-1])<1e-12)|(abs(values[1:])<1e-12))
            def fun(f):
                x,y=self._local(i,f);return (x-ox)*dy-(y-oy)*dx
            for j in candidates:
                lo,hi=self._scan_f[j:j+2]
                if abs(values[j])<1e-12:f=lo
                elif abs(values[j+1])<1e-12:f=hi
                else:f=brentq(fun,lo,hi,xtol=1e-13,rtol=1e-13)
                x,y=self._local(i,f);t=(x-ox)*dx+(y-oy)*dy
                tx,ty=self._local(i,f,1)
                if t<=t_min or abs(tx*dy-ty*dx)<1e-10:continue
                normal=(ty/hypot(tx,ty),-tx/hypot(tx,ty))
                u=(self._cumulative[i]+f*(self._cumulative[i+1]-self._cumulative[i]))%1
                hits.append(BoundaryHit(float(t),(float(x),float(y)),normal,float(u),float(self.arclength_coordinates(u)),float(self.curvature(u))))
        hits.sort(key=lambda h:h.t);out=[]
        for h in hits:
            if not out or abs(h.t-out[-1].t)>=1e-7:out.append(h)
        return out
