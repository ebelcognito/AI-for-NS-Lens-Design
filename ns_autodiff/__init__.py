"""ns_autodiff -- a differentiable PyTorch twin of the non-sequential JS ray tracer.

This package is the "full scoped autodiff" companion to ns_probe/refine_lm.py (which
refines QD-discovered folds with a *finite-difference* Jacobian + Levenberg-Marquardt
through the JS engine). Here the same forward model is reimplemented as **differentiable
torch arithmetic over a frozen, champion-recorded interaction graph**, so the residual
Jacobian comes from reverse-mode autograd instead of 2*n_opt finite-difference traces.

It is deliberately ISOLATED: it imports the existing JS engine ONLY as a read-only
forward-truth oracle (via PersistentRayOpticsRunner) and never edits anything under
ray-optics/ or ns_probe/. See README.md for the design and validation results.
"""
