# Outbound-only CODE V job bridge

This bridge is deliberately **not agent-to-agent chat**.  The remote Ubuntu host writes a
small declarative JSON job; one ordinary process on the Windows CODE V machine polls it
over SFTP, runs a locally approved adapter, and returns JSON plus optional artifacts.  No
OpenAI API call is made by the bridge, so polling and file transfer consume **zero model
tokens**.

The current implementation is ready for the transport test.  The final CODE V adapter
must be completed on the Windows machine, where its actual installed command-line
interface can be inspected and tested.

The short prompt for that machine's agent is in `LOCAL_AGENT_HANDOFF.md` so the full
architecture does not need to be pasted into every conversation.

## The three pieces

1. **Tailscale** supplies private reachability.  Only Windows initiates a connection.
2. **OpenSSH SFTP on Ubuntu** supplies the queue; the account has no shell.
3. **`python -m codev_bridge worker` on Windows** is the persistent poller.  It accepts
   only `codev.run_staged_case` jobs whose case ID and SHA-256 are in its local config.

There is no cloud-side daemon.  An agent or human publishes a job with one CLI command.
The Windows worker can run continuously without either agent being active.

## Security boundary

- Jobs cannot contain scripts, executable paths, `.seq` contents, shell commands,
  environment variables, or arbitrary command-line arguments.  Unknown JSON fields are
  rejected.
- A job identifies an already-staged local case by `case_id` and SHA-256.  The file itself
  is synchronized to Windows separately, inspected locally, and approved in
  `allowed_cases`.  Adding that path and digest is the approval action; never automate it.
- The adapter executable, working directory, and fixed argument template exist only in
  the Windows config.  Execution uses `shell=False`.
- The worker pins the Ubuntu SSH host key, rejects unknown keys, caps age/time/size, claims
  jobs by same-filesystem rename, and publishes the result JSON last.
- The SFTP account is chrooted, public-key-only, non-admin, and has no shell or forwarding.
- Evaluations are **at least once** after a crash.  Only side-effect-free CODE V analysis
  operations belong on this bridge.

## Phase 1 — enroll the physical Windows workstation as a user device

Use the official Windows installer and log in interactively with your Tailscale identity.
An approved administrator must handle the UAC prompt.  Do not give an agent an admin
password or authentication key.  **Do not apply `tag:codev-worker` to this workstation.**
Tailscale tags are intended for non-human service devices, and applying one removes the
device's user identity.

After login, record the non-secret device name and Tailscale IPv4 address:

```powershell
& "$env:ProgramFiles\Tailscale\tailscale.exe" version
& "$env:ProgramFiles\Tailscale\tailscale.exe" status
& "$env:ProgramFiles\Tailscale\tailscale.exe" ip -4
```

The policy below gives only that exact Windows device access to the tagged Ubuntu server.

## Phase 2 — configure the Ubuntu endpoint (manual, reviewed commands)

Do this only after approval from the owner/administrator of this Ubuntu host.  Keep an
existing administrative SSH session open until `sshd -t` and the first SFTP test pass.

Install the stable Ubuntu 24.04 package from Tailscale's repository:

```bash
sudo mkdir -p --mode=0755 /usr/share/keyrings
curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/noble.noarmor.gpg \
  | sudo tee /usr/share/keyrings/tailscale-archive-keyring.gpg >/dev/null
curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/noble.tailscale-keyring.list \
  | sudo tee /etc/apt/sources.list.d/tailscale.list >/dev/null
sudo apt-get update
sudo apt-get install tailscale
```

Before enrollment, define `tag:nslens-cloud` and the exact Windows IP in the tailnet
policy shown below.  Then enroll Ubuntu interactively—never paste an auth key into either
agent:

```bash
sudo tailscale up \
  --hostname=nslens-cloud \
  --advertise-tags=tag:nslens-cloud \
  --accept-routes=false \
  --accept-dns=false
```

Create the chroot and transfer-only account:

```bash
sudo groupadd --system codevbridge
sudo useradd --system --no-create-home --home-dir /bridge \
  --shell /usr/sbin/nologin --gid codevbridge codevbridge
sudo passwd -l codevbridge
sudo usermod -aG codevbridge hisham26

sudo install -d -o root -g root -m 0755 /srv/codevbridge
sudo install -d -o root -g codevbridge -m 2770 /srv/codevbridge/bridge
sudo install -d -o codevbridge -g codevbridge -m 2770 \
  /srv/codevbridge/bridge/jobs/pending \
  /srv/codevbridge/bridge/jobs/claimed \
  /srv/codevbridge/bridge/results/ready \
  /srv/codevbridge/bridge/archive \
  /srv/codevbridge/bridge/rejected
sudo install -d -o root -g root -m 0755 /etc/ssh/authorized_keys
```

The new group applies to a new login session.  The host path used by the Ubuntu job
publisher is `/srv/codevbridge/bridge`; because of the chroot, the Windows SFTP client
must use `/bridge`.  These two paths name the same directory from opposite sides.

Generate the worker key on Windows under the existing standard Windows account:

```powershell
ssh-keygen -t ed25519 -a 64 -f "$env:USERPROFILE\.ssh\codevbridge_ed25519" -C "codev-worker"
```

For unattended operation, this implementation cannot prompt for a key passphrase.  Use an
unencrypted worker key protected by the current account's NTFS permissions, or extend and
review the worker to use a suitable local SSH agent.  This key is additionally constrained
to the workstation's exact Tailscale address and the chrooted, no-shell account.

Copy **only** `codevbridge_ed25519.pub` to the Ubuntu administrator.  Never copy or paste
the file without `.pub`.  Install its single line in
`/etc/ssh/authorized_keys/codevbridge`, prefixed as follows:

```text
from="100.108.189.42",restrict ssh-ed25519 PUBLIC_KEY_MATERIAL codev-worker
```

Set the file to root ownership and mode 0600.  Append this account-specific block to
`/etc/ssh/sshd_config` using `sudoedit`:

```text
Match User codevbridge
    ChrootDirectory /srv/codevbridge
    ForceCommand internal-sftp -d /bridge -u 0007
    AuthorizedKeysFile /etc/ssh/authorized_keys/%u
    AuthenticationMethods publickey
    PubkeyAuthentication yes
    PasswordAuthentication no
    KbdInteractiveAuthentication no
    GSSAPIAuthentication no
    PermitEmptyPasswords no
    PermitTTY no
    DisableForwarding yes
    X11Forwarding no
    AllowAgentForwarding no
    AllowTcpForwarding no
    PermitTunnel no
    MaxAuthTries 2
    MaxSessions 1
```

Validate, then reload—do not restart—SSH:

```bash
sudo /usr/sbin/sshd -t
sudo systemctl reload ssh
```

The tailnet policy must be merged with the existing policy and broader grants audited;
grants are additive:

```json
{
  "tagOwners": {
    "tag:nslens-cloud": []
  },
  "hosts": {
    "codev-worker": "100.108.189.42"
  },
  "grants": [
    {
      "src": ["codev-worker"],
      "dst": ["tag:nslens-cloud"],
      "ip": ["tcp:22"]
    }
  ],
  "tests": [
    {
      "src": "codev-worker",
      "proto": "tcp",
      "accept": ["tag:nslens-cloud:22"],
      "deny": ["tag:nslens-cloud:23"]
    },
    {
      "src": "tag:nslens-cloud",
      "proto": "tcp",
      "deny": ["codev-worker:22"]
    }
  ]
}
```

The `codev-worker` alias is pinned to the enrolled `HIRES103` address.  The tests require
Windows-to-Ubuntu TCP 22, reject a neighboring port, and fail the policy update if any
broader rule permits Ubuntu-to-Windows TCP 22.  `tag:nslens-cloud` remains appropriate for
Ubuntu because it is the service endpoint.  Remove the default allow-all grant; grants are
additive and a narrower grant cannot override it.

## Phase 3 — verify SFTP manually

On Ubuntu, obtain the public host-key fingerprint locally:

```bash
sudo ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
tailscale ip -4
```

On Windows, connect to the displayed Tailscale IP and compare the fingerprint before
accepting it into `known_hosts`:

```powershell
sftp -i "$env:USERPROFILE\.ssh\codevbridge_ed25519" codevbridge@100.x.y.z
```

Inside SFTP, `pwd` should show `/bridge`; `ls` should show `jobs`, `results`, `archive`,
and `rejected`.  A shell login must fail.

## Phase 4 — run the harmless mock worker

The reported Windows folder is a standalone copy at `C:\Users\hisham26\codev`, not a
Git clone or an installation of the whole project.  That is fine: run the package from
its parent directory and install only its SFTP dependency.  The pinned Paramiko release
supports Python 3.13.

First copy the current `codev_bridge/` folder and
`codev_handoff/benchmark_tierA_v2c.seq` into that project root through your existing,
reviewed file-transfer route.  This is one-time staging, not a bridge job.  Then run:

```powershell
$ProjectRoot = "C:\Users\hisham26\codev"
$BridgeHome = Join-Path $env:LOCALAPPDATA "NSLensBridge"
$Python = Join-Path $env:LOCALAPPDATA "Programs\Python\Python313\python.exe"

New-Item -ItemType Directory -Force -Path $BridgeHome
& $Python -m venv (Join-Path $BridgeHome "venv")
$VenvPython = Join-Path $BridgeHome "venv\Scripts\python.exe"
& $VenvPython -m pip install -r (Join-Path $ProjectRoot "codev_bridge\requirements-windows.txt")
Copy-Item (Join-Path $ProjectRoot "codev_bridge\worker.example.json") (Join-Path $BridgeHome "worker.json")
```

Edit `%LOCALAPPDATA%\NSLensBridge\worker.json` locally:

- set the Ubuntu Tailscale IP and the Windows username in the two key paths;
- set the absolute path and SHA-256 of the local staged file;
- leave `"adapter": {"type": "mock"}` for the first test.

Calculate the staged file hash without sending the file:

```powershell
(Get-FileHash .\codev_handoff\benchmark_tierA_v2c.seq -Algorithm SHA256).Hash.ToLowerInvariant()
```

Start the worker in the foreground:

```powershell
Set-Location C:\Users\hisham26\codev
& "$env:LOCALAPPDATA\NSLensBridge\venv\Scripts\python.exe" -m codev_bridge worker `
  --config "$env:LOCALAPPDATA\NSLensBridge\worker.json" `
  --log-file "$env:LOCALAPPDATA\NSLensBridge\worker.log"
```

On Ubuntu, publish the matching harmless job:

```bash
python3 -m codev_bridge submit \
  --root /srv/codevbridge/bridge \
  --case-id tier_a_v2c \
  --case-file codev_handoff/benchmark_tierA_v2c.seq
```

The worker should return a result saying that connectivity passed and CODE V was not
started.  Check it with the emitted job UUID:

```bash
python3 -m codev_bridge status \
  --root /srv/codevbridge/bridge JOB_UUID
```

## Phase 5 — connect the trusted CODE V adapter

The Windows preflight found two installed automation routes.  Prefer the documented
`CodeV.Application` COM interface over `codev.exe /B`: COM exposes synchronous command
output, warnings/errors/license events, timeouts, and an explicit `StopCodeV()`, while the
installed help does not promise meaningful `/B` process exit codes.  The COM adapter can
use the same Python 3.13 environment after installing its Windows-only dependency:

```powershell
$ProjectRoot = "C:\Users\hisham26\codev"
$VenvPython = "$env:LOCALAPPDATA\NSLensBridge\venv\Scripts\python.exe"
& $VenvPython -m pip install -r (Join-Path $ProjectRoot "codev_bridge\requirements-codev-adapter-windows.txt")
```

Do not run this installation or start CODE V until the mock transport succeeds and the
license/desktop-session decision has been reviewed.

The local adapter must accept exactly four locally supplied values:

```text
trusted-adapter.exe --job-file NORMALIZED_JOB_JSON \
                    --case-file LOCALLY_APPROVED_STAGED_FILE \
                    --result-file RESULT_JSON \
                    --artifact-dir EMPTY_OUTPUT_DIRECTORY
```

It maps `tier_a_v2c` to the locally staged, hash-approved prescription, invokes the actual
installed CODE V automation interface, and writes:

```json
{
  "summary": {
    "loaded_without_errors": true,
    "chief_ray_completed": true
  },
  "artifacts": [
    {"name": "error-log.txt", "media_type": "text/plain"},
    {"name": "layout.png", "media_type": "image/png"}
  ]
}
```

Artifact names are basenames; the corresponding files must be created inside the supplied
artifact directory.  Configure the worker only after that adapter has been tested locally:

```json
"adapter": {
  "type": "fixed_command",
  "executable": "C:\\Users\\hisham26\\AppData\\Local\\NSLensBridge\\venv\\Scripts\\python.exe",
  "working_directory": "C:\\Users\\hisham26\\codev",
  "arguments": [
    "-m", "codev_bridge.codev_com_adapter",
    "--job-file", "{job_file}",
    "--case-file", "{case_file}",
    "--result-file", "{result_file}",
    "--artifact-dir", "{artifact_dir}"
  ]
}
```

The exact CODE V executable and flags are intentionally absent from remote jobs.  The
local agent should discover them from the installed version and implement them only in
this adapter.

`codev.run_staged_case` is the safe bootstrap operation, not the final high-throughput
prescription protocol.  Once it works, add a second operation containing a closed,
numeric optical-prescription schema and generate the CODE V macro in the trusted local
adapter.  Do not relax this operation to accept remote `.seq` text or arbitrary fields.

## Phase 6 — keep the worker running

After the mock and one CODE V job both succeed, register the same worker command in
Windows Task Scheduler under the dedicated, non-admin account with:

- trigger: **At log on**;
- run level: **Limited / least privilege**;
- restart on failure: three attempts, one minute apart;
- multiple instances: **Do not start a new instance**;
- executable: `%LOCALAPPDATA%\NSLensBridge\venv\Scripts\pythonw.exe` (replace the
  environment variable with its resolved absolute path in Task Scheduler);
- arguments: `-m codev_bridge worker --config C:\Users\hisham26\AppData\Local\NSLensBridge\worker.json --log-file C:\Users\hisham26\AppData\Local\NSLensBridge\worker.log`;
- working directory: `C:\Users\hisham26\codev`.

Run it interactively for the first few jobs.  Do not create the scheduled task until its
behavior and CODE V license handling are understood.

## Token behavior and limitations

The bridge can poll every five seconds forever without model usage.  It also avoids
re-sending the repository or runbook: jobs are typically under 1 KiB and results can be
compact summaries.  Screenshots and full logs remain artifacts and cost no tokens unless
an agent is asked to inspect them.

An agent itself is not a persistent daemon.  This remote agent will not automatically wake
when a result arrives.  Ask it to check a returned job, or later add a deliberate scheduled
agent/API invocation; that reasoning step necessarily uses tokens.  Do not make the local
agent part of every job once the deterministic adapter works.
