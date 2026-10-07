# Paystack deployment

Eidomira initializes Paystack transactions only on the server. Never expose `STUDIO_PAYSTACK_SECRET_KEY` to browser JavaScript, logs, source control, or support screenshots.

## Dashboard setup

1. Use Paystack Test Mode.
2. Create a monthly NGN plan for ₦59,900 and copy its `PLN_...` code.
3. Create an annual NGN plan for ₦599,000 and copy its `PLN_...` code.
4. Set the webhook URL to `https://YOUR_DOMAIN/api/payments/paystack/webhook`.
5. Set the public application URL in `STUDIO_PUBLIC_URL`.
6. Configure the environment values from `.env.example`.
7. Complete successful, declined, duplicate-webhook, cancellation, and renewal tests.
8. Confirm ledger balances and idempotency before changing from `sk_test_...` to `sk_live_...`.

The amounts in the Eidomira environment must exactly match the Paystack dashboard plans. Amounts use kobo.

## Security behavior

- Checkout requires an authenticated, email-verified user.
- Every checkout uses a server-generated unique reference.
- The callback never grants value by itself; Eidomira calls Paystack's verify endpoint.
- Webhook bodies are verified with HMAC-SHA512 using `x-paystack-signature` before parsing or fulfilment.
- Raw webhook hashes are stored for idempotency.
- Currency and expected amount are checked before credits are granted.
- Top-ups and subscription grants use an auditable ledger reference.
- Paystack keys must come from a secrets manager in production.

Annual checkout grants 18,000 subscription credits for the annual term. Monthly checkout grants 1,500 on each verified renewal. Purchased PAYG credits remain separate and expire 12 months after the latest successful top-up.

## PAYG packs

- 200 credits: ₦12,000
- 500 credits: ₦27,500
- 1,500 credits: ₦75,000
- 5,000 credits: ₦225,000

PAYG users do not receive the daily Voice Changer allowance, priority processing, subscription pricing, or API access. They may use supported interactive tools while their purchased balance is sufficient. Synthetic-media provenance remains mandatory.

## Production requirements

Use PostgreSQL rather than SQLite for horizontally scaled API replicas. Wrap fulfilment in database transactions with row locking, restrict operational database access, configure alerts for webhook failures, and reconcile Paystack settlements against the credit ledger daily.
