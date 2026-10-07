# Third-party code review

Short, factual records of third-party repositories we evaluated and why. The purpose is to
stop the same trap being re-opened six months from now when the search result looks
attractive again.

---

## `Stelepryz/Deep-Live-Cam-User-Face` — **MALICIOUS. Do not clone, open or build.**

**Reviewed:** 2026-10-07 · **Verdict:** weaponised malware lure. No usable code.
**Action:** report to GitHub; never open in Visual Studio or run `dotnet build`.

### What it pretends to be

It borrows the name of the well-known `Deep-Live-Cam` project. The README is keyword-stuffed
with twenty fake topics (`deep-live-cam`, `faceswap`, `virtual-camera-router`,
`deep-fake-stream-api`, …) to hijack GitHub search, and advertises
*"2FA/ID Test Bypass: Simulates user presence for verification."* The actual C# source is
unrelated game-automation code for Axie Infinity (`NosGame.Utils`, OpenCV template matching
against game UI screenshots). A `.github/workflows/main.yml` cron job runs **every minute**
to commit a one-line timestamp and force-push, farming the repo's "recently updated" signal.

### The payload

`BypassCam/BypassCam[.]csproj` carries an MSBuild target that runs before compilation:

```xml
<Target Name="PreBuild" BeforeTargets="PreBuildEvent">
  <Exec Command="powershell.exe -WindowStyle Hidden -EncodedCommand &lt;base64 blob&gt;" />
</Target>
```

The blob is UTF-16LE base64 — the `-EncodedCommand` wire format. It is **not reproduced here**;
decoding it on review yielded a dropper that:

1. writes a second-stage script to `%TEMP%\<GUID>.ps1` and executes it **hidden** with
   `-ExecutionPolicy Bypass`;
2. polls, every 30 seconds forever, two endpoints for the real payload:
   - `hxxp://130[.]12[.]182[.]172/ApiCertificate.txt`
   - `hxxps://muckanthropic[.]com/LG/Api-Certificate`
3. stages it at `C:\Users\Public\Pictures\t[.]ps1` (a world-writable directory — a common
   malware staging location) and runs it hidden;
4. targets the persistence path
   `C:\ProgramData\Windows[.]Microsoft[.]Photos\current\Microsoft[.]exe`, which imitates genuine
   Microsoft/Photos directories;
5. exits early if both `t.ps1` and `Microsoft.exe` already exist, to avoid re-infecting and
   to survive reboots quietly.

`-EncodedCommand` plus `-WindowStyle Hidden` plus a `PreBuild` target means the malware runs
from a plain **build**, which is the exact action the README instructs the reader to take.

### Campaign context

This matches **Operation "Muck and Load"**, a lure network Socket documented in July 2026:
222 confirmed repositories across 190 accounts, Muck-themed infrastructure
(`muckcoding[.]com`, `muckdeveloper[.]com`), the same `Api-Certificate` naming, the same
`C:\Users\Public\Pictures\` staging and the same synthetic commit-farming workflow. The
cluster delivered Vidar infostealer, RAT and spyware droppers, and Monero miners. Sibling
repos using this exact template (`Stellaryz/…-2fa-User-Face`,
`Feefgganapt/Deep-Live-Cam-User-Face`) are still online.

### Remediation if it was opened

Cloning alone does not execute anything. `PreBuild` fires on **build** (Visual Studio build,
`dotnet build`, `msbuild`).

1. Disconnect the machine from the network.
2. Delete `C:\Users\Public\Pictures\t[.]ps1`, `C:\ProgramData\Windows.Microsoft.Photos\`
   and any `%TEMP%\*[.]ps1` you cannot account for.
3. Run a full Microsoft Defender / EDR scan, then verify no Run key or scheduled task points
   at those paths.
4. Rotate everything a stealer would take: browser-saved passwords, GitHub PATs and SSH keys,
   cloud and payment API keys, crypto wallets, session cookies. Revoke GitHub OAuth apps too.
5. Prefer reimaging over cleaning if the payload actually ran.

### Indicators of compromise

| Type | Value |
| --- | --- |
| IP | `130[.]12[.]182[.]172` |
| Domain | `muckanthropic[.]com` |
| Domain family | `muckcoding[.]com`, `muckdeveloper[.]com` |
| URL path | `/LG/Api-Certificate`, `/ApiCertificate.txt` |
| File | `C:\Users\Public\Pictures\t[.]ps1` (staging) |
| File | `C:\ProgramData\Windows[.]Microsoft[.]Photos\current\Microsoft[.]exe` (persistence) |
| Build marker | `-EncodedCommand` inside a `.csproj` `Exec` task |
| Actor email | `ischhfd83[@]rambler[.]ru` (commit-farming workflow) |

### A second, independent reason to reject it

Even setting the malware aside, the advertised capability — defeating liveness and ID
verification — is the direct opposite of Eidomira's own safety boundary: active liveness
challenges and an identity-match gate exist precisely to block static-photo and replay
spoofing. There was never anything here to adopt.

---

## Checks worth running on any repo before we borrow code

1. **Read every build file.** `.csproj`, `.props`, `.targets`, `Makefile`, `setup.py`,
   `package.json` scripts, `.github/workflows/*`. Build hooks execute on build, not on review.
2. **Search for hidden execution:** `EncodedCommand`, `-WindowStyle Hidden`,
   `ExecutionPolicy Bypass`, `Invoke-WebRequest`, `curl | sh`, `certutil -decode`, base64 blobs.
3. **Decode and read anything base64.** It is obfuscation, never convenience.
4. **Match the licence to the project.** This one claimed MIT while linking an unrelated
   project's licence file.
5. **Check whether the code does what the README sells.** Here they had nothing in common.
6. **Prefer provenance.** Zero-star repo, one commit, force-pushing activity every minute is
   a signal, not a small project.
7. **Weigh the feature against our own policy.** A capability that defeats identity
   verification or consent is out of scope for Eidomira regardless of its licence.

---

## `facefusion/facefusion` — **legitimate, but licence-blocked for code reuse**

**Reviewed:** 2026-10-07 · **Verdict:** real, well-built project. We may learn from its
architecture; we may not copy its code or ship its model weights.
**Action:** no code taken. One idea adopted, reimplemented independently (see below).

### Safety check (the checklist, applied)

Clean. All Python, no binaries, no build hooks in the CI workflow, no `eval`/`exec`,
no base64 blobs, no obfuscation. `subprocess` is used only for FFmpeg and conda.
Model downloads are integrity-checked against hash files — note the check is CRC32,
which detects corruption but is **not** tamper-proof, so it is not a supply-chain control.

### Licence — the blocker

| Period | Licence |
| --- | --- |
| ≤ 2024-05-19 | MIT |
| ≥ 2025-02-15 | **OpenRAIL-AS** |

`LICENSE.md` is 51 bytes and states only "OpenRAIL-AS license" — the terms themselves are
not in the repository. OpenRAIL is **not** an OSI-approved open-source licence: it is a
Responsible AI Licence carrying use-based restrictions that must be passed downstream.

Consequences for Eidomira:

- We cannot copy source from the current tree into a commercial product.
- Adopting code from the pre-2025 MIT snapshot is legally murky (the author re-licensed
  deliberately) and reputationally worse. Do not do it.
- **The model weights are licensed separately and are the sharper problem:**
  INSwapper, ArcFace and AlphaFace are listed as **non-commercial**, HyperSwap uses
  ResearchRAIL. Other weights are MIT/Apache/GPL/unknown. This is the same wall our own
  README already documents for `inswapper_128.onnx`.

### What is worth taking — ideas, reimplemented independently

Copyright covers the expression, not the engineering idea. These are standard patterns,
implemented in `app/engines/providers.py` and elsewhere on our own terms:

1. **Auto-detect execution providers instead of hardcoding CUDA→CPU.** *Adopted.* Our
   adapter hardcoded `["CUDAExecutionProvider", "CPUExecutionProvider"]`, so a host with
   ROCm, DirectML, CoreML, OpenVINO or TensorRT silently ran on the CPU and simply looked
   slow, with nothing anywhere saying so. `app/engines/providers.py` now asks ONNX Runtime
   what the host exposes, orders it by expected throughput, reports the result through
   `/api/health` and the studio status line, and keeps TensorRT opt-in because it builds
   an engine on first use.

2. **Track a face across frames by bounding-box overlap and refill gaps.** *Not adopted.*
   Frame-to-frame association means one dropped detection does not flicker or swap
   identity. Our `app/temporal.py` works at the pixel level (motion-aware blending) and
   does no cross-frame identity association. Worth doing; a real change, tracked below.

3. **Pool inference sessions and never construct them per frame.** We already build one
   engine per process, so this is a design we happen to satisfy rather than a gap.

4. **Report the ONNX Runtime version range with known problems.** Their code encodes a
   memory-arena issue specific to CUDA plus ORT versions `> 1.25.1` and `< 1.29.0`. Our
   `onnxruntime-gpu>=1.20,<2` range spans it. It may not affect us — their workaround
   exists because they share sessions between contexts, which we do not — but it is worth
   confirming before a GPU launch rather than discovering under load.

### Deliberate non-goals

FaceFusion ships a content filter to detect and block NSFW material. That belongs to its
model set, not to us, and none of its weights or filtering are in scope here.

### Follow-ups

- [ ] Verify the ORT 1.25.1–1.29.0 arena behaviour against our single-session design.
- [ ] Consider bounding-box face tracking in `app/temporal.py` to survive dropped detections.
- [ ] Keep `EIDOMIRA_PROVIDERS` in the deployment docs so operators can pin a provider.
