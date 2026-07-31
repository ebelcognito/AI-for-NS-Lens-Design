"""Smoke-test the ray-optics Node bridge.

Goals:
1. Confirm subprocess + JSON round-trip works end-to-end.
2. Measure per-call timing (one-shot and persistent runners).
3. Verify a known geometry gives a sensible detector reading.

Scene: collimated horizontal beam (parallel, 50 units tall) hits a
biconvex spherical lens (R1=+50, R2=-50, d=5, n=1.5), centered at x=120.
Detector at x=240 (~focal plane for thin-lens: f = R/(2(n-1)) = 50).
"""

import sys
from pathlib import Path

# Make the optical_rl package importable.
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from optical_rl.utils.ray_optics_bridge import run_simulation
from optical_rl.utils.ray_optics_bridge_persistent import PersistentRayOpticsRunner


def build_scene():
    """Collimated beam -> biconvex lens -> detector."""
    return {
        "version": 5,
        "objs": [
            # Vertical-line emission segment of a parallel beam.
            # Beam direction is perpendicular to (p1 -> p2), so for a horizontal
            # beam we set p1 above and p2 below at the same x.
            {
                "type": "Beam",
                "p1": {"x": 0, "y": -25},
                "p2": {"x": 0, "y":  25},
                "brightness": 0.5,
                "emisAngle": 0.0,
                "lambert": False,
                "random": False,
            },
            # Biconvex thin-ish lens. f = R / (2(n-1)) = 50 / (2*0.5) = 50.
            {
                "type": "SphericalLens",
                "defBy": "DR1R2",
                "p1": {"x": 120, "y": -25},
                "p2": {"x": 120, "y":  25},
                "params": {"d": 5.0, "r1": 50.0, "r2": -50.0},
                "refIndex": 1.5,
                "cauchyB": 0.004,
                "partialReflect": False,
            },
            # Vertical detector at the geometric focal plane.
            {
                "type": "Detector",
                "p1": {"x": 240, "y": -10},
                "p2": {"x": 240, "y":  10},
                "irradMap": True,
                "binSize": 1.0,
            },
        ],
    }


def summarize(label, parsed, timing):
    print(f"\n--- {label} ---")
    if parsed.get("error"):
        print(f"  ERROR: {parsed['error']}")
        return
    if parsed.get("warning"):
        print(f"  warn:  {parsed['warning']}")
    print(f"  python total:        {timing['total_ms']:8.2f} ms")
    print(f"    json_encode:       {timing['json_encode_ms']:8.2f} ms")
    print(f"    subprocess:        {timing['subprocess_ms']:8.2f} ms")
    print(f"    json_decode:       {timing['json_decode_ms']:8.2f} ms")
    js = parsed.get("timing") or {}
    if js:
        print(f"  js total:            {js.get('total_ms', 0):8.2f} ms")
        print(f"    scene_create:      {js.get('scene_create_ms', 0):8.2f} ms")
        print(f"    scene_load:        {js.get('scene_load_ms', 0):8.2f} ms")
        print(f"    simulator_create:  {js.get('simulator_create_ms', 0):8.2f} ms")
        print(f"    simulation_run:    {js.get('simulation_run_ms', 0):8.2f} ms")
        print(f"    detector_extract:  {js.get('detector_extract_ms', 0):8.2f} ms")
    print(f"  processedRayCount:   {parsed.get('processedRayCount')}")
    print(f"  totalTruncation:     {parsed.get('totalTruncation')}")
    for i, det in enumerate(parsed.get("detectors", [])):
        print(f"  detector[{i}] {det.get('name')!r}:")
        print(f"    power            = {det['power']:.6f}")
        print(f"    normal           = {det['normal']:.6f}")
        print(f"    shear            = {det['shear']:.6f}")
        if det.get("irradianceMap"):
            irr = det["irradianceMap"]
            pos = det["binPositions"]
            peak_idx = max(range(len(irr)), key=lambda i: irr[i])
            print(f"    irradiance bins  = {len(irr)}")
            print(f"    peak bin idx     = {peak_idx} (pos={pos[peak_idx]:.2f}, val={irr[peak_idx]:.4f})")
            # Print a tiny ASCII profile around the peak.
            lo = max(0, peak_idx - 4)
            hi = min(len(irr), peak_idx + 5)
            for i in range(lo, hi):
                bar = "#" * int(40 * irr[i] / max(irr[peak_idx], 1e-12))
                marker = " <-peak" if i == peak_idx else ""
                print(f"      pos {pos[i]:6.2f}: {irr[i]:7.4f}  {bar}{marker}")


def main():
    scene = build_scene()

    # ---- one-shot runner: subprocess per call ----
    print("=" * 60)
    print("ONE-SHOT RUNNER (subprocess per call)")
    print("=" * 60)
    parsed, timing = run_simulation(scene, verbose=False)
    summarize("one-shot, call 1", parsed, timing)

    # second call to see the cold/warm spread.
    parsed2, timing2 = run_simulation(scene, verbose=False)
    summarize("one-shot, call 2", parsed2, timing2)

    # ---- persistent runner: shared Node process ----
    print("\n" + "=" * 60)
    print("PERSISTENT RUNNER (shared Node process)")
    print("=" * 60)
    with PersistentRayOpticsRunner(verbose=False) as runner:
        runner.start()
        for i in range(5):
            result = runner.simulate(scene, episode=0, step=i)
            # PersistentRayOpticsRunner.simulate returns just the parsed dict
            # (or a (parsed, timing) tuple depending on version).
            if isinstance(result, tuple):
                parsed, timing = result
            else:
                parsed = result
                timing = {
                    "total_ms": 0.0,
                    "json_encode_ms": 0.0,
                    "subprocess_ms": 0.0,
                    "json_decode_ms": 0.0,
                }
            summarize(f"persistent, call {i+1}", parsed, timing)


if __name__ == "__main__":
    main()
