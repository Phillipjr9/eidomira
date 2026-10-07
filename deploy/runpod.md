# RunPod deployment

1. Create a Secure Cloud Pod with an RTX 4090 or L40S, 30 GB container disk, and a persistent volume.
2. Expose HTTP port `8000` through RunPod's HTTPS proxy.
3. Build this repository as a Docker image and configure the pod to run it.
4. Mount a legitimately licensed model at `/models/inswapper_128.onnx`.
5. Set:

```env
STUDIO_BACKEND=inswapper
STUDIO_MODEL_PATH=/models/inswapper_128.onnx
STUDIO_REQUIRE_SELF_VERIFICATION=true
```

6. Deploy Coturn from `deploy/docker-compose.turn.yml` on a host with a public IP and open UDP/TCP 3478, TLS 5349, and the configured relay range.
7. Set `STUDIO_TURN_URLS`, `STUDIO_TURN_SECRET`, and `STUDIO_TURN_REALM` to match Coturn. Never expose the shared TURN secret to the browser; the API generates expiring credentials.
8. Verify `/api/health` returns `"gpu": true` and `"turn_configured": true`.

For public production, place a regional WebRTC gateway in front of the worker. The MVP uses binary WebSocket frames so it remains easy to test and deploy; this transport should be replaced by WebRTC before high-concurrency launch.
