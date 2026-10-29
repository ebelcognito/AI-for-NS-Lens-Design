# Decision after `LOCAL_BRIDGE_PREFLIGHT`

The Windows preflight passes the architectural checks and identifies no fundamental
blocker.  Use the installed `CodeV.Application` COM automation interface for the final
adapter.  Do not use process exit status from `codev.exe /B` as the optical-success
signal.

The following must be resolved before a live mock connection:

1. Obtain the relevant administrator/IT approval to install Tailscale on the UNC-managed
   Ubuntu and Windows machines and to enroll them in the chosen tailnet.
2. Keep the physical Windows workstation user-owned in Tailscale; do not apply
   `tag:codev-worker`.  Restrict the grant to its exact Tailscale IPv4/host alias.  Tag only
   the Ubuntu service endpoint as `tag:nslens-cloud`.
3. Copy the latest `codev_bridge/` directory and
   `codev_handoff/benchmark_tierA_v2c.seq` to `C:\Users\hisham26\codev`.  Verify that the
   SHA-256 of the staged `.seq` matches on Ubuntu and Windows.
4. Use the Python-3.13-specific, user-writable setup commands in `README.md`; the `py`
   launcher and `C:\NSLensBridge` are not assumed anymore.

The user selected the existing physical workstation and reported authorization to proceed.
Enroll Windows first so its exact Tailscale IP can be inserted into the policy and SSH-key
source restriction.  Then perform the Ubuntu/SFTP setup and run the mock
adapter.  The mock must complete before installing pywin32 or starting CODE V.

After the mock succeeds, the next local-agent task is:

> Read the installed CODE V 2026 COM help and vendor event example already identified in
> `LOCAL_BRIDGE_PREFLIGHT`. Implement `codev_bridge/codev_com_adapter.py` against the
> existing fixed-adapter contract, using `CodeV.Application`, `StartingDirectory`,
> `StartCodeV()`, `Session.Command()`, and `StopCodeV()` in `finally`. Capture CODE V,
> warning, and license-error events; `LIS`; `GSC`; and only locally hard-coded trace/layout
> commands. Never accept an executable, CODE V command, path, or argument from job JSON.
> For this turn use a fake COM object for tests and do not start CODE V or consume a
> license. Return the proposed one-run live validation command separately for approval.
