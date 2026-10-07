# Mobile calls and camera limitations

## What works

- Install Eidomira as a PWA on iOS or Android.
- Run a live transformation inside Eidomira.
- Record processed video with synchronized local microphone audio.
- Share the resulting clip through the mobile share sheet.
- Screen-share the Eidomira PWA in calling apps that support mobile screen sharing.

## What mobile operating systems normally block

A web app cannot register a virtual camera for WhatsApp, FaceTime, Instagram, TikTok, or another native app. iOS does not expose a general third-party virtual-camera API. Standard Android applications also cannot replace the system camera exposed to another app without privileged/OEM access, rooting, or modified system components.

## Practical live-call workflows

1. Use the call application's screen-sharing feature and share Eidomira's clean-output view.
2. Join the call from a desktop and use OBS Virtual Camera.
3. Build calling directly into a future native Eidomira application using a service such as LiveKit, Daily, or Agora. That supports transformed video inside calls hosted by Eidomira, but still does not turn Eidomira into WhatsApp's camera.

Never install unofficial root-level virtual-camera packages on a personal phone without understanding their security implications.
