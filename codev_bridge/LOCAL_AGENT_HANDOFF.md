# Read-only handoff for the agent on the Windows CODE V machine

We are establishing an outbound-only bridge.  The Windows machine will poll a restricted
Ubuntu SFTP account over Tailscale.  Remote jobs are strict declarative JSON and may only
name a locally staged, SHA-256-approved case.  They never supply code, `.seq` contents,
paths, executables, or command-line arguments.

Read `codev_bridge/README.md`, `codev_bridge/worker.example.json`, and the files in the
local `codev_bridge/` package.  For this turn perform **only a read-only preflight**.  Do
not install anything, enroll Tailscale, generate keys, change firewall/SSH settings,
create users/tasks/services, start a persistent worker, consume a CODE V license, load a
design, or modify the repository.

Return a section titled `LOCAL_BRIDGE_PREFLIGHT` containing:

1. Windows edition/version/build and whether this is a physical workstation or VM.
2. Current username, whether it is administrator, and whether a dedicated standard
   account or VM can run CODE V with the required license.
3. Tailscale installed/version/status/device identity, without displaying auth keys or
   other secrets.
4. OpenSSH `ssh`/`sftp` availability and versions.
5. Python launcher/interpreter availability and versions.
6. Absolute path of this repository and whether `codev_bridge/` is present.
7. SHA-256 of `codev_handoff/benchmark_tierA_v2c.seq` (a hash is non-secret).
8. Installed CODE V release/build and license mode, without displaying credentials or a
   license-server address.
9. The exact installed CODE V automation entry point supported on this machine: program
   name/path, version/help source, whether it can run a command-language `.seq` file,
   whether it requires an interactive desktop session, how success/failure is signaled,
   and how Error Log, lens data, global coordinates, trace summary, and a layout image can
   be exported.  Prefer installed help or vendor documentation.  Do not guess flags and
   do not execute a design during this preflight.
10. Any blocker to running one long-lived, non-admin Python poller with outbound-only SFTP.

Do not print passwords, private keys, Tailscale auth keys, CODE V credentials, license
server addresses, or unrelated environment variables.  Do not paste large help files;
summarize them and quote only the exact relevant command syntax.

End with:

```json
{
  "local_device_name": null,
  "local_tailscale_ip": null,
  "project_root": null,
  "staged_case_sha256": null,
  "codev_release": null,
  "codev_automation_entry_point": null,
  "requires_interactive_desktop": null,
  "safe_to_begin_mock_transport_test": false,
  "manual_actions_still_required": []
}
```

Stop after the report.  The mock transport test and CODE V adapter implementation will be
reviewed as separate phases.
