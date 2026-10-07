/* ==========================================================================
   EIDOMIRA LIVE — landing interactions
   Vanilla, no dependencies. Every animation respects prefers-reduced-motion.
   ========================================================================== */
(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  // matchMedia is universally available in browsers but is missing in some embedded
  // webviews — never let a media query take the whole page down.
  const mq = (query) => {
    try {
      return typeof window.matchMedia === "function"
        ? window.matchMedia(query)
        : { matches: false };
    } catch {
      return { matches: false };
    }
  };
  const reduce = mq("(prefers-reduced-motion: reduce)").matches;
  const fine = mq("(hover: hover) and (pointer: fine)").matches;

  /* ---------------------------------------------------------------------
     Auth (unchanged contract: /api/auth/* , token in localStorage)
     --------------------------------------------------------------------- */
  const AUTH = (() => {
    let mode = "register";
    const tokenKey = "eidomira_access_token";

    const el = {
      modal: $("#authModal"),
      form: $("#authForm"),
      title: $("#authTitle"),
      copy: $("#authCopy"),
      email: $("#authEmail"),
      password: $("#authPassword"),
      submit: $("#authSubmit"),
      swap: $("#authSwitch"),
      message: $("#authMessage"),
      close: $("#authClose"),
    };
    if (!el.modal || !el.form) return { open() {}, close() {} };

    function paint() {
      const register = mode === "register";
      el.title.textContent = register ? "Start your free trial" : "Welcome back";
      el.copy.textContent = register
        ? "Verify your email to receive 100 credits for 7 days. No card required."
        : "Sign in to open your private Eidomira workspace.";
      el.submit.textContent = register ? "Create account" : "Sign in";
      el.swap.textContent = register ? "Already registered? Sign in" : "New to Eidomira? Start free";
      el.password.autocomplete = register ? "new-password" : "current-password";
      el.message.textContent = "";
      el.message.style.color = "";
    }

    function open(next = "register") {
      mode = next;
      paint();
      el.modal.hidden = false;
      document.body.classList.add("is-locked");
      requestAnimationFrame(() => el.email.focus());
    }

    function close() {
      el.modal.hidden = true;
      document.body.classList.remove("is-locked");
    }

    el.close?.addEventListener("click", close);
    el.modal.addEventListener("click", (event) => {
      if (event.target === el.modal) close();
    });
    el.swap.addEventListener("click", () => {
      mode = mode === "register" ? "login" : "register";
      paint();
    });

    el.form.addEventListener("submit", async (event) => {
      event.preventDefault();
      el.submit.disabled = true;
      try {
        const response = await fetch(`/api/auth/${mode}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email: el.email.value, password: el.password.value }),
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "Authentication failed");
        if (result.access_token) {
          localStorage.setItem(tokenKey, result.access_token);
          location.href = "/app";
        } else {
          el.message.textContent = result.message;
          el.message.style.color = "#6ee7b7";
        }
      } catch (error) {
        el.message.textContent = error.message;
        el.message.style.color = "#ff8fc4";
      } finally {
        el.submit.disabled = false;
      }
    });

    async function verifyEmail() {
      const token = new URLSearchParams(location.search).get("verify");
      if (!token) return;
      history.replaceState({}, "", location.pathname);
      open("login");
      el.message.textContent = "Verifying your email…";
      try {
        const response = await fetch("/api/auth/verify-email", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ token }),
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "Verification failed");
        localStorage.setItem(tokenKey, result.access_token);
        location.href = "/app";
      } catch (error) {
        el.message.textContent = error.message;
        el.message.style.color = "#ff8fc4";
      }
    }

    return { open, close, verifyEmail };
  })();

  $$("#startBtn, #heroStart, #finalStart, #drawerCta, #footerStart, #faqStart, .priceSignup")
    .forEach((button) => button.addEventListener("click", () => AUTH.open("register")));
  $("#loginBtn")?.addEventListener("click", () => AUTH.open("login"));
  $$("[data-auth]").forEach((link) =>
    link.addEventListener("click", (event) => {
      event.preventDefault();
      AUTH.open(link.dataset.auth);
    })
  );

  if (new URLSearchParams(location.search).get("signin")) AUTH.open("login");
  AUTH.verifyEmail();

  /* ---------------------------------------------------------------------
     Scroll progress + sticky nav
     --------------------------------------------------------------------- */
  const nav = $("#nav");
  const bar = $("#progressBar");
  let ticking = false;

  function onScroll() {
    const y = window.scrollY;
    if (bar) {
      const max = document.documentElement.scrollHeight - window.innerHeight;
      bar.style.width = `${max > 0 ? Math.min(100, (y / max) * 100) : 0}%`;
    }
    nav?.classList.toggle("is-stuck", y > 12);
    ticking = false;
  }

  addEventListener(
    "scroll",
    () => {
      if (!ticking) {
        ticking = true;
        requestAnimationFrame(onScroll);
      }
    },
    { passive: true }
  );
  onScroll();

  /* ---------------------------------------------------------------------
     Mobile drawer
     --------------------------------------------------------------------- */
  const burger = $("#burger");
  burger?.addEventListener("click", () => {
    const open = document.body.classList.toggle("drawer-open");
    burger.setAttribute("aria-expanded", String(open));
  });
  $$("#drawer a").forEach((link) =>
    link.addEventListener("click", () => {
      document.body.classList.remove("drawer-open");
      burger?.setAttribute("aria-expanded", "false");
    })
  );

  /* ---------------------------------------------------------------------
     Cursor spotlight (fine pointers only)
     --------------------------------------------------------------------- */
  if (fine && !reduce) {
    const spot = $("#spotlight");
    document.body.classList.add("has-pointer");
    let sx = innerWidth / 2;
    let sy = 0;
    let cx = sx;
    let cy = sy;
    addEventListener(
      "pointermove",
      (event) => {
        sx = event.clientX;
        sy = event.clientY;
      },
      { passive: true }
    );
    (function follow() {
      cx += (sx - cx) * 0.12;
      cy += (sy - cy) * 0.12;
      if (spot) spot.style.transform = `translate3d(${cx}px, ${cy}px, 0)`;
      requestAnimationFrame(follow);
    })();
  }

  /* ---------------------------------------------------------------------
     Reveal on scroll
     --------------------------------------------------------------------- */
  const revealables = $$("[data-reveal]");
  if ("IntersectionObserver" in window && !reduce) {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          entry.target.classList.add("is-in");
          if (entry.target.classList.contains("stat")) entry.target.classList.add("is-in");
          observer.unobserve(entry.target);
        });
      },
      { rootMargin: "0px 0px -12% 0px", threshold: 0.12 }
    );
    revealables.forEach((node) => observer.observe(node));
  } else {
    revealables.forEach((node) => node.classList.add("is-in"));
  }

  /* ---------------------------------------------------------------------
     Counters
     --------------------------------------------------------------------- */
  const counters = $$("[data-count]");
  if (counters.length) {
    const run = (node) => {
      const target = Number(node.dataset.count) || 0;
      if (reduce) {
        node.textContent = String(target);
        return;
      }
      const duration = 1400;
      const start = performance.now();
      const step = (now) => {
        const t = Math.min(1, (now - start) / duration);
        const eased = 1 - Math.pow(1 - t, 3);
        node.textContent = String(Math.round(target * eased));
        if (t < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    };
    if ("IntersectionObserver" in window) {
      const co = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (!entry.isIntersecting) return;
            run(entry.target);
            co.unobserve(entry.target);
          });
        },
        { threshold: 0.5 }
      );
      counters.forEach((node) => co.observe(node));
    } else {
      counters.forEach(run);
    }
  }

  /* ---------------------------------------------------------------------
     Tilt + shine (specimens, art panels)
     --------------------------------------------------------------------- */
  if (fine && !reduce) {
    $$("[data-tilt]").forEach((node) => {
      const strength = Number(node.dataset.tiltStrength) || 6;
      node.addEventListener("pointermove", (event) => {
        const rect = node.getBoundingClientRect();
        const px = (event.clientX - rect.left) / rect.width;
        const py = (event.clientY - rect.top) / rect.height;
        node.style.setProperty("--ry", `${(px - 0.5) * strength * 2}deg`);
        node.style.setProperty("--rx", `${(0.5 - py) * strength * 2}deg`);
        node.style.setProperty("--gx", `${px * 100}%`);
        node.style.setProperty("--gy", `${py * 100}%`);
      });
      node.addEventListener("pointerleave", () => {
        node.style.setProperty("--ry", "0deg");
        node.style.setProperty("--rx", "0deg");
      });
    });
  }

  /* ---------------------------------------------------------------------
     Magnetic buttons
     --------------------------------------------------------------------- */
  if (fine && !reduce) {
    $$("[data-magnet]").forEach((node) => {
      node.addEventListener("pointermove", (event) => {
        const rect = node.getBoundingClientRect();
        const dx = (event.clientX - (rect.left + rect.width / 2)) / rect.width;
        const dy = (event.clientY - (rect.top + rect.height / 2)) / rect.height;
        node.style.transform = `translate(${dx * 8}px, ${dy * 6}px)`;
      });
      node.addEventListener("pointerleave", () => {
        node.style.transform = "";
      });
    });
  }

  /* ---------------------------------------------------------------------
     Hero parallax
     --------------------------------------------------------------------- */
  const heroBg = $("#heroBg");
  if (heroBg && !reduce) {
    const parallax = () => {
      const y = window.scrollY;
      if (y > innerHeight * 1.3) return;
      heroBg.style.transform = `translate3d(0, ${y * 0.18}px, 0)`;
    };
    addEventListener("scroll", () => requestAnimationFrame(parallax), { passive: true });
  }

  /* ---------------------------------------------------------------------
     Live monitor: signal canvas + telemetry
     --------------------------------------------------------------------- */
  const canvas = $("#signal");
  if (canvas) {
    const ctx = canvas.getContext("2d");
    const dpr = Math.min(2, devicePixelRatio || 1);
    const width = canvas.clientWidth || 900;
    const height = 74;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    let visible = true;
    if ("IntersectionObserver" in window) {
      new IntersectionObserver((entries) => {
        visible = entries[0]?.isIntersecting ?? true;
      }).observe(canvas);
    }

    let t = 0;
    function frame() {
      if (visible) {
        t += 0.016;
        ctx.clearRect(0, 0, width, height);

        // baseline
        ctx.strokeStyle = "rgba(255,255,255,0.06)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, height / 2);
        ctx.lineTo(width, height / 2);
        ctx.stroke();

        // layered signal
        for (let layer = 0; layer < 3; layer++) {
          const amp = [16, 10, 5][layer];
          const speed = [1, 1.6, 2.4][layer];
          const gradient = ctx.createLinearGradient(0, 0, width, 0);
          gradient.addColorStop(0, "rgba(139,108,255,0)");
          gradient.addColorStop(0.35, ["rgba(139,108,255,0.9)", "rgba(82,211,255,0.75)", "rgba(255,143,196,0.5)"][layer]);
          gradient.addColorStop(1, "rgba(82,211,255,0)");
          ctx.strokeStyle = gradient;
          ctx.lineWidth = layer === 0 ? 1.6 : 1;
          ctx.beginPath();
          for (let x = 0; x <= width; x += 4) {
            const p = x / width;
            const env = Math.sin(p * Math.PI);
            const y =
              height / 2 +
              Math.sin(x * 0.028 * speed + t * speed) * amp * env +
              Math.sin(x * 0.09 + t * 1.6 * speed) * (amp / 3) * env;
            x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
          }
          ctx.stroke();
        }
      }
      if (!reduce) requestAnimationFrame(frame);
    }
    if (reduce) frame();
    else requestAnimationFrame(frame);

    // telemetry drifting within believable bands
    const latency = $("#tLatency");
    const fps = $("#tFps");
    if (latency && fps && !reduce) {
      let tick = 0;
      setInterval(() => {
        tick += 1;
        const jitter = Math.sin(tick / 3) * 12;
        latency.textContent = `${Math.round(74 + jitter)} ms`;
        fps.textContent = `${Math.round(29 + Math.sin(tick / 2) * 2)} fps`;
      }, 1100);
    } else if (latency && fps) {
      latency.textContent = "78 ms";
      fps.textContent = "29 fps";
    }
  }

  /* ---------------------------------------------------------------------
     Steps scroll spy
     --------------------------------------------------------------------- */
  const steps = $$(".step");
  const slides = $$(".steps__slide");
  const hud = $("#stepsHud");
  if (steps.length) {
    const activate = (index) => {
      steps.forEach((node, i) => node.classList.toggle("is-active", i === index));
      slides.forEach((node, i) => node.classList.toggle("is-active", i === index));
      if (hud) hud.textContent = `Step 0${index + 1}`;
    };
    if ("IntersectionObserver" in window && !reduce) {
      const so = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting) activate(Number(entry.target.dataset.step));
          });
        },
        { rootMargin: "-45% 0px -45% 0px", threshold: 0 }
      );
      steps.forEach((node) => so.observe(node));
    }
    steps.forEach((node, i) => node.addEventListener("click", () => activate(i)));
  }

  /* ---------------------------------------------------------------------
     Toolkit tabs
     --------------------------------------------------------------------- */
  const tabs = $$('.tabs [role="tab"]');
  const indicator = $("#tabsInd");
  if (tabs.length) {
    const move = (button) => {
      if (!indicator) return;
      indicator.style.width = `${button.offsetWidth}px`;
      indicator.style.transform = `translateX(${button.offsetLeft - 5}px)`;
    };
    const select = (button) => {
      tabs.forEach((node) => {
        const active = node === button;
        node.setAttribute("aria-selected", String(active));
        const panel = document.getElementById(node.getAttribute("aria-controls"));
        if (panel) panel.classList.toggle("is-active", active);
      });
      move(button);
    };
    tabs.forEach((button) =>
      button.addEventListener("click", () => select(button))
    );
    tabs.forEach((button, i) =>
      button.addEventListener("keydown", (event) => {
        if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
        event.preventDefault();
        const next = tabs[(i + (event.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
        next.focus();
        select(next);
      })
    );
    const initial = tabs.find((node) => node.getAttribute("aria-selected") === "true") || tabs[0];
    requestAnimationFrame(() => move(initial));
    addEventListener("resize", () => move(tabs.find((n) => n.getAttribute("aria-selected") === "true") || tabs[0]));
  }

  /* ---------------------------------------------------------------------
     Use-case rail: arrows + drag
     --------------------------------------------------------------------- */
  const rail = $("#rail");
  if (rail) {
    const prev = $("#railPrev");
    const next = $("#railNext");
    const stride = () => Math.min(rail.clientWidth * 0.8, 480);

    const refresh = () => {
      const max = rail.scrollWidth - rail.clientWidth - 4;
      if (prev) prev.disabled = rail.scrollLeft <= 4;
      if (next) next.disabled = rail.scrollLeft >= max;
    };
    prev?.addEventListener("click", () => rail.scrollBy({ left: -stride(), behavior: "smooth" }));
    next?.addEventListener("click", () => rail.scrollBy({ left: stride(), behavior: "smooth" }));
    rail.addEventListener("scroll", refresh, { passive: true });
    addEventListener("resize", refresh);
    refresh();

    let down = false;
    let startX = 0;
    let startLeft = 0;
    rail.addEventListener("pointerdown", (event) => {
      if (event.pointerType === "touch") return;
      down = true;
      startX = event.clientX;
      startLeft = rail.scrollLeft;
      rail.classList.add("is-dragging");
    });
    addEventListener("pointermove", (event) => {
      if (!down) return;
      rail.scrollLeft = startLeft - (event.clientX - startX);
    });
    addEventListener("pointerup", () => {
      if (!down) return;
      down = false;
      rail.classList.remove("is-dragging");
    });
  }

  /* ---------------------------------------------------------------------
     FAQ accordion
     --------------------------------------------------------------------- */
  $$(".qa").forEach((item) => {
    const button = $(".qa__q", item);
    const panel = $(".qa__a", item);
    if (!button || !panel) return;
    button.addEventListener("click", () => {
      const open = item.classList.contains("is-open");
      $$(".qa.is-open").forEach((other) => {
        if (other === item) return;
        other.classList.remove("is-open");
        $(".qa__q", other)?.setAttribute("aria-expanded", "false");
        const otherPanel = $(".qa__a", other);
        if (otherPanel) otherPanel.style.height = "0px";
      });
      item.classList.toggle("is-open", !open);
      button.setAttribute("aria-expanded", String(!open));
      panel.style.height = open ? "0px" : `${panel.scrollHeight}px`;
    });
  });

  /* ---------------------------------------------------------------------
     Billing period switch
     --------------------------------------------------------------------- */
  const monthly = $("#billMonthly");
  const annual = $("#billAnnual");
  const price = $("#proPrice");
  const period = $("#proPeriod");
  const note = $("#proNote");
  if (monthly && annual && price && period && note) {
    const setPeriod = (isAnnual) => {
      monthly.setAttribute("aria-pressed", String(!isAnnual));
      annual.setAttribute("aria-pressed", String(isAnnual));
      price.textContent = isAnnual ? "₦599,000" : "₦59,900";
      period.textContent = isAnnual ? "/ year" : "/ month";
      note.textContent = isAnnual
        ? "Billed yearly — 1,500 credits every month, ≈ ₦49,917/month."
        : "1,500 credits every month, billed monthly. Cancel any time.";
      if (!reduce && price.animate) {
        price.animate(
          [
            { opacity: 0, transform: "translateY(8px)", filter: "blur(6px)" },
            { opacity: 1, transform: "none", filter: "blur(0)" },
          ],
          { duration: 520, easing: "cubic-bezier(0.22,1,0.36,1)" }
        );
      }
    };
    monthly.addEventListener("click", () => setPeriod(false));
    annual.addEventListener("click", () => setPeriod(true));
  }

  /* ---------------------------------------------------------------------
     Demo specimens → open the Studio path
     --------------------------------------------------------------------- */
  $$(".spec").forEach((card) =>
    card.addEventListener("click", () => AUTH.open("register"))
  );
})();
