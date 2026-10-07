/* The owner console renders what the server reports.
 *
 * Every value here comes from the database or the filesystem by way of /api/admin/overview,
 * which means some of it is user-supplied text — an email address someone chose. So nothing
 * is ever interpolated into markup: nodes are built and their textContent set. A console
 * that renders addresses with innerHTML would turn "which account is this" into a stored
 * cross-site scripting bug on the highest-privilege page in the product. The rule is easy to
 * keep because it is absolute: this file contains no innerHTML at all.
 */
"use strict";

const $ = (id) => document.getElementById(id);

/* The session lives in localStorage and travels as a header — the same store and the same
 * header `app.js` uses. It is not read from a cookie, because a console that trusts a cookie
 * is a console that is blank exactly where this one was deployed: the studio's first working
 * deployment served it inside a frame whose cookies never arrived. */
const TOKEN_KEY = "eidomira_access_token";
const token = () => localStorage.getItem(TOKEN_KEY) || "";

async function apiFetch(url, options = {}) {
  const headers = new Headers(options.headers || {});
  const session = token();
  if (session) headers.set("Authorization", "Bearer " + session);
  return fetch(url, { ...options, headers });
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function bytes(n) {
  if (n < 1024) return n + " B";
  if (n < 1024 * 1024) return (n / 1024).toFixed(0) + " KB";
  if (n < 1024 * 1024 * 1024) return (n / 1048576).toFixed(1) + " MB";
  return (n / 1073741824).toFixed(2) + " GB";
}

function naira(kobo) {
  return "\u20a6" + (kobo / 100).toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function when(seconds) {
  if (!seconds) return "—";
  return new Date(seconds * 1000).toLocaleString();
}

function ago(seconds) {
  if (!seconds) return "—";
  const delta = Math.max(0, Math.floor(Date.now() / 1000) - seconds);
  if (delta < 90) return "just now";
  if (delta < 3600) return Math.round(delta / 60) + " min ago";
  if (delta < 86400) return Math.round(delta / 3600) + " h ago";
  if (delta < 86400 * 30) return Math.round(delta / 86400) + " d ago";
  return new Date(seconds * 1000).toLocaleDateString();
}

function row(label, value, tone) {
  const line = el("div", "ad-row");
  line.append(el("span", null, label));
  line.append(el("span", tone || null, value));
  return line;
}

function truth(flag) {
  return flag ? "yes" : "no";
}

function count(value, label) {
  const box = el("div", "ad-count");
  box.append(el("b", null, value));
  box.append(el("span", null, label));
  return box;
}

function fill(id, nodes) {
  const host = $(id);
  host.replaceChildren();
  if (!nodes.length) {
    host.append(el("p", "ad-empty", "Nothing to show."));
    return;
  }
  nodes.forEach((node) => host.append(node));
}

function table(id, headers, rows) {
  const host = $(id);
  host.replaceChildren();
  const head = el("tr");
  headers.forEach((label) => head.append(el("th", null, label)));
  const thead = el("thead");
  thead.append(head);
  const tbody = el("tbody");
  if (!rows.length) {
    const cell = el("td", "ad-empty", "None yet.");
    cell.colSpan = headers.length;
    const line = el("tr");
    line.append(cell);
    tbody.append(line);
  }
  rows.forEach((cells) => {
    const line = el("tr");
    cells.forEach((cell) => {
      const td = el("td", cell && cell.className);
      if (cell && cell.node) td.append(cell.node);
      else td.textContent = cell && cell.text !== undefined ? String(cell.text) : String(cell ?? "—");
      line.append(td);
    });
    tbody.append(line);
  });
  host.append(thead, tbody);
}

function tag(text, kind) {
  return { className: "num", node: el("span", "tag" + (kind ? " tag--" + kind : ""), text) };
}

function showError(message, link) {
  const box = $("error");
  box.replaceChildren(el("strong", null, message));
  if (link) {
    box.append(" ");
    const anchor = el("a", null, link.label);
    anchor.href = link.href;
    box.append(anchor);
  }
  box.hidden = false;
  $("loading").hidden = true;
}

/* ── the verdict ─────────────────────────────────────────────────────────── */

function renderVerdict(system) {
  const host = $("verdict");
  host.replaceChildren();
  host.className = "ad-verdict " + (system.can_swap ? "ad-verdict--ready" : "ad-verdict--blocked");

  const headline = system.can_swap
    ? "This installation can run a neural face swap."
    : "This installation cannot run a neural face swap.";
  host.append(el("h1", null, headline));

  const copy = system.can_swap
    ? "The configured weights are present and the runtime that loads them is installed. " +
      "What visitors receive still depends on the quality stack behind the engine, not on this page."
    : "Reported from the filesystem and the installed runtimes — not from configuration. A " +
      "setting that claims otherwise changes nothing, so each reason is listed rather than summarised.";
  host.append(el("p", null, copy));

  if (system.blockers.length) {
    const list = el("ul");
    system.blockers.forEach((reason) => list.append(el("li", null, reason)));
    host.append(list);
  }

  const serving = el("p", null, "");
  serving.style.marginTop = "14px";
  serving.append("The engine currently serving visitors is ");
  serving.append(el("code", null, system.engine.name));
  serving.append(system.engine.name === "diagnostic"
    ? " — the path that validates a pipeline without a neural model."
    : ".");
  host.append(serving);
}

/* ── panels ──────────────────────────────────────────────────────────────── */

function renderEngine(system) {
  const engine = system.engine;
  const settings = system.settings;
  fill("engine", [
    row("Engine", engine.name, engine.name === "diagnostic" ? "warn" : "good"),
    row("Execution provider", engine.provider || "—", engine.provider ? "good" : "dim"),
    row("Providers available", engine.providers.length ? engine.providers.join(", ") : "none", engine.providers.length ? null : "dim"),
    row("Hardware acceleration", truth(engine.accelerated), engine.accelerated ? "good" : "dim"),
    row("Pixel boost", settings.swap_pixel_boost + "\u00d7", settings.swap_pixel_boost > 1 ? "good" : "dim"),
    row("Max frame width", settings.max_frame_width + " px"),
    row("Restoration model", settings.restoration_model_path,
        system.weights.restoration_present ? "good" : "dim"),
    row("Restoration visibility", settings.restoration_visibility),
    row("Tone transfer", settings.tone_transfer_strength),
    row("Temporal stabiliser", settings.temporal_strength),
    row("Trainer", settings.trainer_enabled ? "enabled" : "off", settings.trainer_enabled ? "good" : "dim"),
  ]);
}

function renderModels(system) {
  const weights = system.weights;
  $("weightsNote").textContent =
    "Weight files found in " + weights.directory +
    ". Nothing downloads these: a model is either in the repository or it is not.";

  const rows = [
    row("Configured swap model", weights.configured_swap_present ? "present" : "missing",
        weights.configured_swap_present ? "good" : "bad"),
    row("Face parser", weights.parser_present ? "present" : "missing", weights.parser_present ? "good" : "dim"),
    row("Restoration model", weights.restoration_present ? "present" : "missing",
        weights.restoration_present ? "good" : "dim"),
  ];
  Object.entries(system.runtimes).forEach(([name, installed]) => {
    rows.push(row(name, installed ? "installed" : "not installed", installed ? "good" : "bad"));
  });
  fill("models", rows);

  table("weights", ["File", "Size", "Modified"],
    weights.swap.map((w) => [w.name, bytes(w.bytes), when(w.modified)]));
}

function renderAccounts(data) {
  const counts = data.counts;
  fill("accountCounts", [
    count(data.total, "accounts"),
    count(counts.admins, "admins"),
    count(counts.demo, "demo"),
    count(counts.paying, "paying"),
    count(counts.verified, "verified"),
    count(counts.unverified, "unverified"),
    count(counts.disabled, "disabled"),
  ]);

  table("accounts", ["Email", "Role", "Plan", "Credits", "Used", "Joined", "State"],
    data.accounts.map((a) => [
      { className: "email", node: el("span", null, a.email) },
      { className: "num", node: el("span", "tag" + (a.role === "admin" ? " tag--admin" : (a.demo ? " tag--warn" : "")),
                                   a.role + (a.demo ? " \u00b7 demo" : "")) },
      a.plan ? (a.plan_active ? a.plan : a.plan + " (" + a.plan_status + ")") : "—",
      { className: "num", text: a.credits },
      { className: "num", text: a.credits_used },
      { className: "num", text: ago(a.created_at) },
      a.disabled ? tag("disabled", "off") : (a.verified ? tag("verified", "paid") : tag("unverified", "warn")),
    ]));

  $("accountNote").textContent = data.truncated
    ? "Showing the " + data.shown + " most recent of " + data.total + " accounts."
    : "Registration is open to anyone who can reach the site. Administrators are made with " +
      "python -m tools.grant_admin <email> on the server — never through the site, and never by " +
      "a default account shipped in the repository.";
}

function renderPayments(data) {
  const statuses = Object.entries(data.by_status);
  fill("paymentCounts", [
    count(statuses.reduce((sum, [, v]) => sum + v.count, 0), "intents"),
    count(data.events_received, "webhook events"),
    count(data.webhooks_unprocessed, "unprocessed"),
  ].concat(statuses.map(([status, v]) => count(naira(v.kobo), status))));

  table("paymentTable", ["When", "Account", "Product", "Amount", "Status"],
    data.recent.map((p) => [
      { className: "num", text: ago(p.created_at) },
      { className: "email", node: el("span", null, p.email || "—") },
      p.product,
      { className: "num", text: naira(p.amount_kobo) },
      { className: "num", node: el("span", "tag" + (p.status === "success" ? " tag--paid" : ""), p.status) },
    ]));

  $("paymentNote").textContent = data.recent.length
    ? "Amounts are shown as recorded, in kobo converted for display only. Unprocessed webhooks " +
      "mean Paystack delivered an event the application has not yet applied."
    : "No payment has been attempted on this installation. Checkout needs a Paystack secret key " +
      "before it can be.";
}

function renderLive(data, system) {
  fill("live", [
    row("Identity sessions", data.sessions + " of " + data.session_capacity,
        data.sessions === 0 ? "dim" : "good"),
    row("WebRTC peers", data.peers + " of " + data.peer_capacity, data.peers === 0 ? "dim" : "good"),
    row("Session lifetime", Math.round(system.settings.session_ttl_seconds / 60) + " min"),
    row("Public URL", system.settings.public_url),
  ]);
}

function renderSecurity(system) {
  const security = system.security;
  const rows = [
    security.auth_secret_is_repository_default
      ? row("Session signing key", "repository default — anyone who has read the repo can forge a session", "bad")
      : row("Session signing key", "set by this deployment", "good"),
    row("Accounts required", truth(security.require_auth),
        security.require_auth ? "good" : "warn"),
    row("Self-verification", truth(security.require_self_verification)),
    row("Access token lifetime", Math.round(security.access_token_ttl / 60) + " min"),
    row("Requests per minute", security.rate_limit_per_minute),
    row("Sign-in attempts per hour", security.login_limit_per_hour),
    row("Session enrollments per hour", security.enrollment_limit_per_hour),
    row("Email delivery", security.email_configured ? "configured" : "not configured",
        security.email_configured ? "good" : "warn"),
    row("Paystack", security.paystack_configured ? "configured" : "not configured",
        security.paystack_configured ? "good" : "warn"),
    row("One-click demo sign-in", security.demo_login ? "ON — anyone who can reach this server can use it" : "off",
        security.demo_login ? "bad" : "dim"),
    row("TURN relay", security.turn_configured ? "configured" : "not configured",
        security.turn_configured ? "good" : "dim"),
    row("Calls (LiveKit)", security.calls_configured ? "configured" : "not configured",
        security.calls_configured ? "good" : "dim"),
    row("Database", system.settings.database_path),
  ];
  fill("security", rows);
}

function renderReports(data) {
  $("reportNote").textContent = data.count
    ? data.count + " report(s) in " + data.directory + ". Written by the trainer, one per session."
    : "No reports yet in " + data.directory + ". The trainer writes one per session it observes.";
  table("reports", ["Report", "Size", "Written"],
    data.latest.map((r) => [r.name, bytes(r.bytes), when(r.modified)]));
}

/* ── boot ────────────────────────────────────────────────────────────────── */

async function load() {
  const response = await apiFetch("/api/admin/overview", { headers: { Accept: "application/json" } });
  if (response.status === 401) {
    showError("You are not signed in on this browser.", { label: "Sign in", href: "/?signin=1" });
    return;
  }
  if (response.status === 403) {
    showError("This account is not an administrator. Access is granted on the server with " +
              "python -m tools.grant_admin <email>.", { label: "Back to the studio", href: "/app" });
    return;
  }
  if (!response.ok) {
    showError("The console could not read this installation (HTTP " + response.status + ").");
    return;
  }

  const data = await response.json();
  renderVerdict(data.system);
  renderEngine(data.system);
  renderModels(data.system);
  renderAccounts(data.accounts);
  renderPayments(data.payments);
  renderLive(data.live, data.system);
  renderSecurity(data.system);
  renderReports(data.reports);

  $("stamp").textContent = "read at " + when(data.generated_at) + " · " + window.location.host;
  $("loading").hidden = true;
  $("console").hidden = false;
}

async function whoAmI() {
  try {
    const me = await apiFetch("/api/auth/me", { headers: { Accept: "application/json" } });
    if (!me.ok) return;
    const user = await me.json();
    $("who").textContent = user.email || "—";
  } catch { /* the header is decoration; the console content is not */ }
}

$("signOut").addEventListener("click", async () => {
  try { await apiFetch("/api/auth/logout", { method: "POST" }); } catch { /* leaving anyway */ }
  localStorage.removeItem(TOKEN_KEY);
  window.location.href = "/";
});

whoAmI();
load().catch((error) => showError("The console could not load: " + error.message));
