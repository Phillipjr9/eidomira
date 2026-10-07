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
