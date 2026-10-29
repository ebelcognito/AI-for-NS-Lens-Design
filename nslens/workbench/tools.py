"""Stateful tools with immutable requirements, exact tracing budgets, and logs."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json

import numpy as np

from nslens.learning.freeform_repair import ImagingSpec, merit
from .domain import COEFFICIENT_BOUND_MM, PARAMETERS, diagnostics, measure

SCHEMA = "nslens.optical_workbench.v1"


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def tool(name, description, properties, required=None):
    return dict(type="function", name=name, description=description, strict=True,
                parameters=dict(type="object", properties=properties,
                                required=list(properties) if required is None else required,
                                additionalProperties=False))


TOOL_SCHEMAS = [
    tool("inspect", "Read current coefficients, active freedoms, field/pupil diagnostics and remaining budget. Uses cached coarse measurement.", {}),
    tool("enable", "Set the editable polynomial terms on one or both optical regions. Does not change coefficients or trace rays.", {
        "surface": dict(type="string", enum=["entrance", "exit", "both"]),
        "powers": dict(type="array", items=dict(type="integer", enum=[2, 3, 4]), minItems=1, maxItems=3)}),
    tool("perturb", "Apply a measured coefficient change. Costs one coarse trace; physical failures are rejected. A worse but physical design may become current.", {
        "parameter": dict(type="string", enum=list(PARAMETERS)), "delta_mm": dict(type="number", minimum=-.5, maximum=.5)}),
    tool("optimize", "Optimize only enabled coefficients with measured finite differences and guarded steps. The max_evaluations allowance includes ALL internal probes. At least active_count+1 calls are needed.", {
        "max_evaluations": dict(type="integer", minimum=2, maximum=24)}),
    tool("restore_best", "Restore the best physically ranked coarse state already visited, without another trace.", {}),
    tool("verify", "Check the current design with 119 rays across 7 fields. Limited to two checks; final hidden audit uses 279 rays and is not exposed.", {}),
    tool("finish", "Submit the current design. Results are independently re-traced after the episode; a textual claim alone is not success.", {
        "summary": dict(type="string", maxLength=1000)}),
]


class Workbench:
    def __init__(self, task, budget=24, max_actions=16):
        self.task = json.loads(json.dumps(task, allow_nan=False))
        # Public task schema deliberately has no witness/teacher answer channel.
        if set(task) != {"task_id", "family_id", "split", "spec", "start", "objective"}:
            raise ValueError("unexpected public task field")
        self.spec = ImagingSpec(**task["spec"])
        if (self.spec.family_id, self.spec.split) != (task["family_id"], task["split"]):
            raise ValueError("family identity mismatch")
        if not isinstance(budget, int) or budget < 1 or not isinstance(max_actions, int) or max_actions < 1:
            raise ValueError("positive integer budgets required")
        self.x = np.asarray(task["start"], dtype=float)
        self.budget, self.max_actions = budget, max_actions
        self.used = self.actions = self.verifications = 0
        self.enabled = []
        self.done = False
        self._final = None
        self.records, self.events = [], []
        self.current = self._measure(self.x, "coarse", "initial")
        self.best_x, self.best = self.x.copy(), self.current

    def _measure(self, x, fidelity, source):
        if fidelity == "coarse":
            if source != "initial" and self.used >= self.budget:
                raise ValueError("coarse candidate budget exhausted")
            self.used += int(source != "initial")
        v = measure(self.spec, x, fidelity)
        self.records.append(dict(index=len(self.records), source=source, fidelity=fidelity,
                                 coefficients=np.asarray(x).tolist(), verdict=asdict(v)))
        return v

    def state(self):
        return dict(coefficients_mm=dict(zip(PARAMETERS, self.x.tolist())),
                    enabled=[PARAMETERS[i] for i in self.enabled],
                    diagnostics=diagnostics(self.current),
                    candidate_calls_used=self.used, candidate_calls_remaining=self.budget-self.used,
                    verification_calls_remaining=2-self.verifications,
                    actions_remaining=max(0, self.max_actions-self.actions), done=self.done)

    def _accept(self, x, v, improving_only):
        if v.physical_valid and (not improving_only or merit(v) < merit(self.current)):
            self.x, self.current = x.copy(), v
        if merit(v) < merit(self.best):
            self.best_x, self.best = x.copy(), v

    def _optimize(self, allowance):
        if not self.enabled:
            raise ValueError("enable surface terms before optimizing")
        if allowance < len(self.enabled) + 1:
            raise ValueError("allowance must cover active_count Jacobian probes plus a correction")
        if allowance > self.budget - self.used:
            raise ValueError("allowance exceeds remaining candidate budget")
        stop = self.used + allowance
        while stop - self.used >= len(self.enabled) + 1:
            base, v0 = self.x.copy(), self.current
            cols = []
            for i in self.enabled:
                h = .002 if base[i] < COEFFICIENT_BOUND_MM-.002 else -.002
                probe = base.copy()
                probe[i] += h
                v = self._measure(probe, "coarse", "jacobian")
                cols.append((np.asarray(v.residual) - v0.residual) / h)
                if merit(v) < merit(self.best):
                    self.best_x, self.best = probe.copy(), v
            jac = np.array(cols).T
            delta = np.linalg.solve(jac.T@jac + .001*np.eye(len(self.enabled)),
                                    -jac.T@np.asarray(v0.residual))
            delta *= min(1., .5 / max(np.linalg.norm(delta), 1e-12))
            for scale in (1., .5, .25):
                if self.used == stop:
                    break
                x = base.copy()
                x[self.enabled] = np.clip(base[self.enabled] + scale*delta, -2., 2.)
                v = self._measure(x, "coarse", "optimization")
                self._accept(x, v, improving_only=True)
                if merit(self.current) < merit(v0):
                    break
        # Unspent allowance is reported; never pad budgets with fictitious traces.

    def call(self, name, arguments):
        before = self.state()
        if self.done or self.actions >= self.max_actions:
            return dict(error="episode finished or action limit reached", state=before)
        self.actions += 1
        try:
            schemas = {s["name"]: s for s in TOOL_SCHEMAS}
            if name not in schemas or not isinstance(arguments, dict):
                raise ValueError("unknown tool or invalid argument object")
            properties = schemas[name]["parameters"]["properties"]
            if set(arguments) != set(properties):
                raise ValueError("missing or unexpected argument")
            for key, value in arguments.items():
                rule = properties[key]
                kind = rule["type"]
                if kind == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
                    raise ValueError("integer argument required")
                if kind == "number" and (isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value)):
                    raise ValueError("finite numeric argument required")
                if kind == "string" and not isinstance(value, str):
                    raise ValueError("string argument required")
                if "enum" in rule and value not in rule["enum"]:
                    raise ValueError("unsupported argument value")
                if "minimum" in rule and value < rule["minimum"] or "maximum" in rule and value > rule["maximum"]:
                    raise ValueError("argument outside bounds")
                if "maxLength" in rule and len(value) > rule["maxLength"]:
                    raise ValueError("text argument too long")
            if name == "enable":
                powers = arguments["powers"]
                if not isinstance(powers, list) or not 1 <= len(powers) <= 3 or any(type(p) is not int or p not in (2,3,4) for p in powers):
                    raise ValueError("powers must be a list drawn from 2, 3, 4")
                side = arguments["surface"]
                self.enabled = [i for i, p in enumerate(PARAMETERS)
                                if (side == "both" or p.startswith(side)) and int(p[-1]) in powers]
            elif name == "perturb":
                i = PARAMETERS.index(arguments["parameter"])
                if i not in self.enabled:
                    raise ValueError("parameter is not enabled")
                x = self.x.copy()
                x[i] += arguments["delta_mm"]
                if np.any(np.abs(x) > 2.):
                    raise ValueError("coefficient outside bounds")
                v = self._measure(x, "coarse", "perturb")
                self._accept(x, v, improving_only=False)
            elif name == "optimize":
                self._optimize(arguments["max_evaluations"])
            elif name == "restore_best":
                self.x, self.current = self.best_x.copy(), self.best
            elif name == "verify":
                if self.verifications >= 2:
                    raise ValueError("verification allowance exhausted")
                self.verifications += 1
                v = self._measure(self.x, "dense", "agent_verification")
                output = dict(verification=asdict(v), state=self.state())
                return self._event(name, arguments, before, output)
            elif name == "finish":
                self.done = True
            output = dict(state=self.state())
        except (ValueError, TypeError) as exc:
            output = dict(error=str(exc), state=self.state())
        return self._event(name, arguments, before, output)

    def _event(self, name, arguments, before, output):
        self.events.append(dict(tool=name, arguments=arguments, before_hash=canonical_hash(before),
                                output=output, after_hash=canonical_hash(self.state())))
        return output

    def finalize(self):
        if self._final is not None:
            return self._final
        submitted = self.done
        self.done = True
        audit = self._measure(self.x, "audit", "hidden_final_audit")
        self._final = dict(schema_version=SCHEMA, task_id=self.task["task_id"],
                    family_id=self.spec.family_id, split=self.spec.split, submitted=submitted,
                    success=bool(submitted and audit.success), final_audit=asdict(audit),
                    final_coefficients=self.x.tolist(), candidate_calls=self.used,
                    initial_calls=1, verification_calls=self.verifications, actions=self.actions,
                    nominal_ray_evaluations=sum(r["verdict"]["rays"] for r in self.records),
                    invalid_tool_calls=sum("error" in e["output"] for e in self.events))
        return self._final
