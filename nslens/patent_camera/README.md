# Patent camera: spherical chief-ray approximation

Run from the repository root:

```sh
.venv/bin/python -m nslens.patent_camera.run
.venv/bin/python -m unittest discover -s tests -p test_patent_camera.py
```

Outputs go to `results/patent_camera_spherical/`: `layout.png`, `layout.svg`,
`report.md`, `chief_rays.json`, and `prescription_spherical.json`.

The experiment uses the existing Python non-sequential Fresnel/TIR engine in
`nslens/freeform_body/trace.py`. `model.py` supplies analytic spherical or polynomial
sag intersections; it does not impose the desired surface sequence. All branches
are traced subject to the recorded limits, and the dominant exit is evaluated.

`prescription.json` retains the source Figure 3A prescription and table text.
`run.spherical_prescription()` drops all aspheric/freeform terms, converts the
XY surfaces' Y2 curvature to a spherical radius, and explicitly shifts P2S1 by
1.5 mm along its local sag axis to make a separate closed prism. This shift is an
approximation choice, not published data. Lens and prism indices are the patent's
d-line values. The IR filter is omitted. Coated mirrors are ideal lossless mirrors.

The native calculation uses (patent Z, patent Y), mm and degrees. Chief rays pass
through the stop center. The plotted sensor span uses the image semi-diagonal.
Five chiefs reaching the sensor verifies the folded path, not focus, f/2 aperture
performance, full-field throughput, image quality, or MTF.

The full polynomial adapter is retained for future work, but the zero-filled
published freeform prescription fails the disjoint-body check: Table 3I does not
list ZP1, ZP3, ZP4, and the corresponding zero-filled CODE V interpretation overlaps
the prisms. The spherical run is independent of those coefficients.
