"""Bounded tools for freeform TIR-prism repair; layouts and routing gates are fixed."""
from __future__ import annotations
import copy
import numpy as np
from .tools import tool,canonical_hash
from .folded_domain import evaluate,PARAMETERS,REQUIREMENTS,LAYOUT

INSTRUCTIONS='''You are repairing a folded freeform AR/VR prism. Three display points emit
full angular fans toward a closed freeform element, which must route light by total
internal reflection into a fixed eyebox. Useful light MUST have at least one TIR;
straight-through paths cannot pass. Requirements: each field folded throughput >=0.30,
angular standard deviation <=1 degree, ordered fields, FOV >=12 degrees, closed energy
accounting, negligible unresolved rays. Surface T is the top region, D bottom, A left,
B right; a region can be visited multiple times. Shape uses endpoint-tapered radial
polynomials on four regions of a single closed C2 spline. Base geometry, material, sources
and pupil are fixed. Choose which regions/orders to change; inspect actual route and
field diagnostics, perturb or optimize, verify, and submit with finish. Optimize performs
measured coordinate search and charges every trial, including rejected ones. There are
up to 48 candidate traces (the task may specify fewer), 16 actions, two optional dense checks and a hidden final audit.
When task constraints are supplied, only listed coefficients are editable; changes from
the starting prescription must stay within max_delta_mm. Enable cannot include locked terms.
This is a folded repair task, not unconstrained topology discovery. No reference answer
is available. Prefer a physically verified outcome to a verbal claim.'''

TOOLS=[
 tool('inspect','Inspect current shape, folded routes, all-field throughput/spread, and remaining budgets.',{}),
 tool('enable','Choose editable freeform regions and polynomial orders. Replaces prior selection.',{
  'faces':dict(type='array',items=dict(type='string',enum=['A','B','T','D']),minItems=1,maxItems=4),
  'powers':dict(type='array',items=dict(type='integer',enum=[2,4]),minItems=1,maxItems=2)}),
 tool('perturb','Try a coefficient change on an enabled term; costs one candidate trace. Physical changes become current even if worse.',{
  'parameter':dict(type='string',enum=list(PARAMETERS)),'delta_mm':dict(type='number',minimum=-1.,maximum=1.)}),
 tool('optimize','Measured coordinate search on enabled terms. Tests +/- step, retains improvements, halves step after a non-improving sweep. ALL probes charged.',{
  'max_evaluations':dict(type='integer',minimum=2,maximum=48),'step_mm':dict(type='number',minimum=.01,maximum=1.)}),
 tool('restore_best','Restore the best coarse prescription visited.',{}),
 tool('verify','Trace the current prism with a denser shifted 99-ray grid; at most two checks.',{}),
 tool('finish','Submit current prescription for hidden 183-ray acceptance; prose is not a success label.',{'summary':dict(type='string',maxLength=1000)})]


class FoldedWorkbench:
 @staticmethod
 def validate_task(task):
  if set(task)-{'constraints'}!={'task_id','family_id','split','base','start','requirements','layout'}:raise ValueError('unexpected public task fields')
  if task['requirements']!=REQUIREMENTS or task['layout']!=LAYOUT:raise ValueError('requirements/layout mismatch')
 def __init__(self,task):
  self.validate_task(task)
  self.task=copy.deepcopy(task);self.x=np.asarray(task['start'],dtype=float)
  if self.x.shape!=(8,) or not np.isfinite(self.x).all() or np.max(np.abs(self.x))>2:raise ValueError('invalid start')
  constraints=task.get('constraints',dict(editable=list(PARAMETERS),candidate_budget=48,max_delta_mm=4.))
  if set(constraints)!={'editable','candidate_budget','max_delta_mm'}:raise ValueError('invalid constraints')
  editable=constraints['editable'];budget=constraints['candidate_budget'];delta=constraints['max_delta_mm']
  if not isinstance(editable,list) or not editable or any(p not in PARAMETERS for p in editable) or len(set(editable))!=len(editable):raise ValueError('invalid editable terms')
  if type(budget) is not int or not 2<=budget<=48:raise ValueError('invalid candidate budget')
  if type(delta) not in (int,float) or not np.isfinite(delta) or not 0<delta<=4:raise ValueError('invalid change limit')
  self.allowed={PARAMETERS.index(p) for p in editable};self.budget=budget
  self.lower=np.maximum(-2,self.x-delta);self.upper=np.minimum(2,self.x+delta)
  self.used=0;self.actions=0;self.verifications=0;self.enabled=[];self.done=False
  self.records=[];self.events=[];self._final=None
  self.current=self.measure(self.x,'coarse','initial');self.best=copy.deepcopy(self.current);self.best_x=self.x.copy()
 def measure(self,x,fidelity,source):
  if fidelity=='coarse' and source!='initial':
   if self.used>=self.budget:raise ValueError('candidate budget exhausted')
   self.used+=1
  v=self.evaluate(x,fidelity)
  self.records.append(dict(coefficients=np.asarray(x).tolist(),fidelity=fidelity,source=source,verdict=v))
  return v
 def evaluate(self,x,fidelity):
  return evaluate(self.task['base'],x,fidelity)
 def state(self):
  return dict(coefficients_mm=dict(zip(PARAMETERS,self.x.tolist())),enabled=[PARAMETERS[i] for i in self.enabled],
   diagnostics=self.current,candidate_calls_remaining=self.budget-self.used,verification_calls_remaining=2-self.verifications,
   actions_remaining=max(0,16-self.actions),done=self.done)
 def accept(self,x,v,only_improving=True):
  if v['physical_valid'] and (not only_improving or not self.current['physical_valid'] or v['merit']<self.current['merit']):self.x,self.current=x.copy(),v
  if v['physical_valid'] and (not self.best['physical_valid'] or v['merit']<self.best['merit']):self.best_x,self.best=x.copy(),copy.deepcopy(v)
 def optimize(self,allowance,step):
  if not self.enabled:raise ValueError('enable surface terms first')
  if allowance>self.budget-self.used:raise ValueError('allowance exceeds remaining budget')
  stop=self.used+allowance
  while self.used<stop:
   old=self.current['merit']
   for i in self.enabled:
    origin=self.x.copy()
    for sign in [1,-1]:
     if self.used==stop:return
     candidate=origin.copy();candidate[i]=np.clip(candidate[i]+sign*step,self.lower[i],self.upper[i])
     v=self.measure(candidate,'coarse','coordinate_trial');self.accept(candidate,v)
   if self.current['merit']>=old:step*=.5
 def call(self,name,args):
  before=canonical_hash(self.state())
  if self.done or self.actions>=16:return dict(error='episode ended',state=self.state())
  self.actions+=1
  try:
   schema=next((t for t in TOOLS if t['name']==name),None)
   if schema is None or not isinstance(args,dict) or set(args)!=set(schema['parameters']['properties']):raise ValueError('invalid tool arguments')
   for key,value in args.items():
    rule=schema['parameters']['properties'][key];kind=rule['type']
    if kind=='integer' and type(value) is not int:raise ValueError('integer required')
    if kind=='number' and (type(value) not in [float,int] or not np.isfinite(value)):raise ValueError('finite number required')
    if kind=='string' and not isinstance(value,str):raise ValueError('string required')
    if kind=='array':
     if not isinstance(value,list) or not rule['minItems']<=len(value)<=rule['maxItems']:raise ValueError('invalid selection array')
     if any(type(v) is not (int if rule['items']['type']=='integer' else str) or v not in rule['items']['enum'] for v in value):raise ValueError('unsupported selection')
    if 'enum' in rule and value not in rule['enum']:raise ValueError('unsupported argument')
    if 'minimum' in rule and not rule['minimum']<=value<=rule['maximum']:raise ValueError('out of bounds')
    if 'maxLength' in rule and len(value)>rule['maxLength']:raise ValueError('text too long')
   if name=='enable':
    selection=[i for i,p in enumerate(PARAMETERS) if p[0] in args['faces'] and int(p[-1]) in args['powers']]
    if not set(selection)<=self.allowed:raise ValueError('selection includes task-locked coefficient')
    self.enabled=selection
   elif name=='optimize':self.optimize(args['max_evaluations'],args['step_mm'])
   elif name=='perturb':
    i=PARAMETERS.index(args['parameter'])
    if i not in self.enabled:raise ValueError('locked coefficient')
    x=self.x.copy();x[i]+=args['delta_mm']
    if not self.lower[i]<=x[i]<=self.upper[i]:raise ValueError('coefficient outside bounds')
    self.accept(x,self.measure(x,'coarse','perturb'),False)
   elif name=='restore_best':self.x,self.current=self.best_x.copy(),copy.deepcopy(self.best)
   elif name=='finish':self.done=True
   elif name=='verify':
    if self.verifications>=2:raise ValueError('verification budget exhausted')
    self.verifications+=1
    result=dict(verification=self.measure(self.x,'dense','verification'),state=self.state())
    return self.event(name,args,before,result)
   result=dict(state=self.state())
  except (ValueError,TypeError) as error:result=dict(error=str(error),state=self.state())
  return self.event(name,args,before,result)
 def event(self,name,args,before,result):
  self.events.append(dict(tool=name,arguments=args,before_hash=before,output=result,after_hash=canonical_hash(self.state())))
  return result
 def finalize(self):
  if self._final is not None:return self._final
  submitted=self.done;self.done=True
  audit=self.measure(self.x,'audit','hidden_audit')
  self._final=dict(submitted=submitted,success=bool(submitted and audit['success']),final_audit=audit,
   final_coefficients=self.x.tolist(),candidate_calls=self.used,verification_calls=self.verifications,actions=self.actions,
   invalid_tool_calls=sum('error' in e['output'] for e in self.events),
   nominal_rays=sum(r['verdict']['rays'] for r in self.records))
  return self._final
