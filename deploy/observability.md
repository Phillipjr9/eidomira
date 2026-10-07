# Observability and privacy

`GET /metrics` exposes Prometheus metrics for HTTP request counts and latency plus active identity sessions and WebRTC peers. Labels intentionally exclude user IDs, room names, IP addresses, identity IDs, filenames, and biometric values.

Protect `/metrics` at the reverse proxy so it is available only to the monitoring network. Suggested initial alerts:

- API 5xx rate above 2% for 5 minutes
- WebRTC setup failures above 10%
- GPU inference p80 above the selected quality target
- Dropped-frame ratio above 20%
- Active peers at configured capacity
- Enrollment 429 rate spike

Do not add raw SDP, camera frames, embeddings, consent documents, access tokens, or email addresses to logs.
