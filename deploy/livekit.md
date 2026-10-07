# Eidomira calls with LiveKit

## Local prototype

1. Copy `livekit.yaml.example` to `livekit.yaml`.
2. Generate credentials:

```bash
API_KEY=$(openssl rand -hex 12)
API_SECRET=$(openssl rand -hex 32)
```

3. Replace the key and secret placeholders in `livekit.yaml`.
4. Set the same values in the Eidomira environment:

```env
STUDIO_LIVEKIT_URL=ws://127.0.0.1:7880
STUDIO_LIVEKIT_API_KEY=...
STUDIO_LIVEKIT_API_SECRET=...
```

5. Start calls infrastructure:

```bash
docker compose -f deploy/docker-compose.calls.yml up -d
```

## Public deployment

Use `wss://` behind TLS, open TCP 7881 and the UDP media range, and configure a public TURN service. Do not expose API secrets to browser code. Eidomira issues short-lived, room-scoped JWTs from the backend. Add account authentication and rate limiting before public access.

For the under-$100 prototype, run LiveKit, Redis, and Coturn on one small VM. Keep neural GPU workers scale-to-zero and route transformed tracks into calls only while a user is actively live.
