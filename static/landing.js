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
     On-device lab: real camera + locally computed face mesh
     The engine and model are only fetched after an explicit click, every
     failure path degrades to copy rather than a broken panel, and the
     camera track is always released on stop.
     --------------------------------------------------------------------- */
  const lab = $("#lab");
  if (lab) {
    const video = $("#labVideo");
    const canvas = $("#labCanvas");
    const idle = $("#labIdle");
    const startBtn = $("#labStart");
    const stopBtn = $("#labStop");
    const status = $("#labStatus");
    const badge = $("#labBadge");
    const studioBtn = $("#labStudio");
    const modeButtons = [$("#labModeMesh"), $("#labModeContour"), $("#labModePoints")];
    const ctx = canvas.getContext("2d");

    const readouts = {
      points: $("#labPoints"),
      fps: $("#labFps"),
      ms: $("#labMs"),
      blinkL: $("#labBlinkL"),
      blinkR: $("#labBlinkR"),
      smile: $("#labSmile"),
      blinkLBar: $("#labBlinkLBar"),
      blinkRBar: $("#labBlinkRBar"),
      smileBar: $("#labSmileBar"),
    };

    // Pinned engine. The first entry is the version whose API these calls were written
    // against; the second is a fallback if the CDN is unreachable or the tag is pulled.
    const ENGINE_CANDIDATES = [
      { module: "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/vision_bundle.mjs", wasm: "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/wasm" },
      { module: "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/vision_bundle.mjs", wasm: "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm" },
    ];
    const MODEL = "/models/face_landmarker.task";

    let landmarker = null;
    let engineApi = null; // the loaded module's FaceLandmarker class, for its static guides
    let pxScale = 1;      // canvas device pixels per CSS pixel
    let lastStamp = 0;    // MediaPipe requires strictly increasing timestamps
    let stream = null;
    let raf = 0;
    let mode = "mesh";
    let lastVideoTime = -1;
    let lastInferMs = 0;
    let fpsSmooth = 0;
    let lastFpsStamp = 0;
    let stopped = true;

    const say = (message, kind = "") => {
      if (!status) return;
      status.innerHTML = message;
      status.className = `lab__status${kind ? ` is-${kind}` : ""}`;
    };

    function setMode(next) {
      mode = next;
      modeButtons.forEach((button, index) => {
        const names = ["mesh", "contour", "points"];
        button?.setAttribute("aria-pressed", String(names[index] === next));
      });
    }

    function sizeCanvas() {
      const rect = canvas.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      const dpr = Math.min(2, devicePixelRatio || 1);
      const width = Math.round(rect.width * dpr);
      const height = Math.round(rect.height * dpr);
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      pxScale = width / rect.width;
    }

    /** Draws one connection list as a polyline in normalised coordinates. */
    function path(landmarks, connections, scaleX, scaleY) {
      for (const { start, end } of connections) {
        const a = landmarks[start];
        const b = landmarks[end];
        if (!a || !b) continue;
        ctx.moveTo(a.x * scaleX, a.y * scaleY);
        ctx.lineTo(b.x * scaleX, b.y * scaleY);
      }
    }

    function draw(result) {
      const { width, height } = canvas;
      const s = pxScale || 1;
      ctx.clearRect(0, 0, width, height);

      const faces = result?.faceLandmarks || [];
      if (!faces.length) return;

      const lm = faces[0];
      const api = engineApi;
      const gradient = ctx.createLinearGradient(0, 0, width, height);
      gradient.addColorStop(0, "rgba(139,108,255,0.95)");
      gradient.addColorStop(0.55, "rgba(82,211,255,0.85)");
      gradient.addColorStop(1, "rgba(255,143,196,0.75)");

      if (mode === "points") {
        ctx.fillStyle = gradient;
        for (const point of lm) {
          ctx.beginPath();
          ctx.arc(point.x * width, point.y * height, 1.5 * s, 0, Math.PI * 2);
          ctx.fill();
        }
      } else {
        const tesselation = api?.FACE_LANDMARKS_TESSELATION;
        if (mode === "mesh" && tesselation) {
          ctx.strokeStyle = "rgba(182,163,255,0.28)";
          ctx.lineWidth = 1 * s;
          ctx.beginPath();
          path(lm, tesselation, width, height);
          ctx.stroke();
        }

        ctx.strokeStyle = gradient;
        ctx.lineWidth = 1.6 * s;
        ctx.lineJoin = "round";
        ctx.beginPath();
        const detail = [
          api?.FACE_LANDMARKS_CONTOURS,
          api?.FACE_LANDMARKS_LEFT_EYE,
          api?.FACE_LANDMARKS_RIGHT_EYE,
          api?.FACE_LANDMARKS_LEFT_EYEBROW,
          api?.FACE_LANDMARKS_RIGHT_EYEBROW,
          api?.FACE_LANDMARKS_LIPS,
        ];
        for (const list of detail) if (list) path(lm, list, width, height);
        ctx.stroke();
      }

      // Iris accents — the points a live session tracks most closely.
      ctx.fillStyle = "rgba(110,231,183,0.95)";
      for (const index of [468, 473, 477, 159, 386]) {
        const point = lm[index];
        if (!point) continue;
        ctx.beginPath();
        ctx.arc(point.x * width, point.y * height, 2.6 * s, 0, Math.PI * 2);
        ctx.fill();
      }

      // Tracking reticle around the face
      let minX = 1, minY = 1, maxX = 0, maxY = 0;
      for (const point of lm) {
        if (point.x < minX) minX = point.x;
        if (point.x > maxX) maxX = point.x;
        if (point.y < minY) minY = point.y;
        if (point.y > maxY) maxY = point.y;
      }
      const pad = 0.045;
      const x0 = Math.max(0, minX - pad) * width;
      const y0 = Math.max(0, minY - pad) * height;
      const x1 = Math.min(1, maxX + pad) * width;
      const y1 = Math.min(1, maxY + pad) * height;
      const arm = Math.min(26 * s, (x1 - x0) * 0.22);
      ctx.strokeStyle = "rgba(110,231,183,0.75)";
      ctx.lineWidth = 2 * s;
      ctx.beginPath();
      ctx.moveTo(x0, y0 + arm); ctx.lineTo(x0, y0); ctx.lineTo(x0 + arm, y0);
      ctx.moveTo(x1 - arm, y0); ctx.lineTo(x1, y0); ctx.lineTo(x1, y0 + arm);
      ctx.moveTo(x1, y1 - arm); ctx.lineTo(x1, y1); ctx.lineTo(x1 - arm, y1);
      ctx.moveTo(x0 + arm, y1); ctx.lineTo(x0, y1); ctx.lineTo(x0, y1 - arm);
      ctx.stroke();
    }

    /** Reads named blendshape scores into the meter rows. */
    function meter(categories) {
      const find = (name) => categories.find((c) => c.categoryName === name)?.score ?? 0;
      const left = find("eyeBlinkLeft");
      const right = find("eyeBlinkRight");
      const smile = Math.max(find("mouthSmileLeft"), find("mouthSmileRight"), find("jawOpen") * 0.6);
      const paint = (bar, label, value) => {
        if (bar) bar.style.width = `${Math.min(100, value * 100)}%`;
        if (label) label.textContent = `${Math.round(value * 100)}%`;
      };
      paint(readouts.blinkLBar, readouts.blinkL, left);
      paint(readouts.blinkRBar, readouts.blinkR, right);
      paint(readouts.smileBar, readouts.smile, smile);
    }

    function loop() {
      if (stopped) return;
      raf = requestAnimationFrame(loop);
      if (!landmarker || video.readyState < 2 || !video.videoWidth) return;
      if (video.currentTime === lastVideoTime) return;
      lastVideoTime = video.currentTime;

      sizeCanvas();
      const started = performance.now();
      lastStamp = Math.max(Math.floor(started), lastStamp + 1);
      let result = null;
      try {
        result = landmarker.detectForVideo(video, lastStamp);
      } catch {
        return;
      }
      lastInferMs = lastInferMs ? lastInferMs * 0.8 + (performance.now() - started) * 0.2 : performance.now() - started;

      draw(result);

      const categories = result?.faceBlendshapes?.[0]?.categories;
      if (categories) meter(categories);

      if (readouts.fps) {
        const now = performance.now();
        const delta = now - (lastFpsStamp || now);
        lastFpsStamp = now;
        if (delta > 0) {
          const instant = 1000 / delta;
          fpsSmooth = fpsSmooth ? fpsSmooth * 0.85 + instant * 0.15 : instant;
          readouts.fps.textContent = `${Math.round(Math.min(fpsSmooth, 99))} fps`;
        }
      }
      if (readouts.ms) readouts.ms.textContent = `${Math.round(lastInferMs)} ms`;
      if (readouts.points) readouts.points.textContent = result?.faceLandmarks?.length ? String(result.faceLandmarks[0].length) : "—";
    }

    async function loadEngine() {
      let lastError = null;
      for (const candidate of ENGINE_CANDIDATES) {
        try {
          const vision = await import(/* webpackIgnore: true */ candidate.module);
          const fileset = await vision.FilesetResolver.forVisionTasks(candidate.wasm);
          const options = {
            baseOptions: { modelAssetPath: MODEL, delegate: "GPU" },
            runningMode: "VIDEO",
            numFaces: 1,
            outputFaceBlendshapes: true,
          };
          try {
            engineApi = vision.FaceLandmarker;
            return await vision.FaceLandmarker.createFromOptions(fileset, options);
          } catch {
            // GPU delegate is unavailable on some drivers — retry on CPU
            options.baseOptions.delegate = "CPU";
            return await vision.FaceLandmarker.createFromOptions(fileset, options);
          }
        } catch (error) {
          lastError = error;
        }
      }
      throw lastError || new Error("Engine unavailable");
    }

    function release() {
      stopped = true;
      cancelAnimationFrame(raf);
      raf = 0;
      stream?.getTracks().forEach((track) => track.stop());
      stream = null;
      video.srcObject = null;
      lab.classList.remove("is-live");
      canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
      if (stopBtn) stopBtn.hidden = true;
      if (startBtn) {
        startBtn.disabled = false;
        startBtn.textContent = "Enable camera";
      }
      modeButtons.forEach((button) => { if (button) button.disabled = true; });
      if (readouts.fps) readouts.fps.textContent = "— fps";
      if (readouts.ms) readouts.ms.textContent = "— ms";
      if (readouts.points) readouts.points.textContent = "—";
    }

    stopBtn?.addEventListener("click", () => {
      release();
      say("Camera released. <b>Nothing was uploaded and no frames were stored.</b>");
      idle?.removeAttribute("aria-hidden");
    });

    studioBtn?.addEventListener("click", () => AUTH.open("register"));
    modeButtons.forEach((button, index) => {
      button?.addEventListener("click", () => setMode(["mesh", "contour", "points"][index]));
    });

    startBtn?.addEventListener("click", async () => {
      if (!navigator.mediaDevices?.getUserMedia) {
        say("This browser doesn't expose a camera. Try Chrome, Edge, Firefox or Safari on a device with a webcam.", "error");
        return;
      }
      startBtn.disabled = true;
      say("Requesting camera access…", "busy");

      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
      } catch (error) {
        startBtn.disabled = false;
        const denied = error?.name === "NotAllowedError" || error?.name === "SecurityError";
        // A blocked prompt inside an embedded frame is almost always the parent's
        // Permissions-Policy, not the visitor's browser setting — say so, and offer the
        // one action that actually fixes it.
        const framed = (() => { try { return window.self !== window.top; } catch { return true; } })();
        const escape = ' <a href="' + location.href + '" target="_blank" rel="noopener">Open in a new tab ↗</a>';
        if (denied && framed) {
          say("Camera is blocked inside this embedded frame. Browsers only grant camera access to a top-level page — press enable again from a new tab." + escape, "error");
        } else if (denied) {
          say("Camera blocked. Allow camera access for this site in your browser's address bar, then press enable again.", "error");
        } else if (!window.isSecureContext) {
          say("Camera access needs a secure (https) connection. " + escape, "error");
        } else {
          say("No camera found. Connect a webcam or try this on a device with one.", "error");
        }
        return;
      }

      video.srcObject = stream;
      try {
        await video.play();
      } catch {
        /* autoplay of a user-gesture stream can still resolve slowly; the loop waits for data */
      }

      // Match the stage to the source so the overlay needs no crop compensation.
      const applyRatio = () => {
        if (video.videoWidth && video.videoHeight) {
          lab.style.setProperty("--ar", `${video.videoWidth} / ${video.videoHeight}`);
        }
      };
      applyRatio();
      video.addEventListener("loadedmetadata", applyRatio, { once: true });

      lab.classList.add("is-live");
      if (stopBtn) stopBtn.hidden = false;
      idle?.setAttribute("aria-hidden", "true");

      if (landmarker) {
        stopped = false;
        lastVideoTime = -1;
        modeButtons.forEach((button) => { if (button) button.disabled = false; });
        say("<b>Tracking on your device.</b> Move your head, blink, smile — the mesh follows.");
        raf = requestAnimationFrame(loop);
        return;
      }

      say("Downloading the on-device model (~10 MB, cached after the first run)…", "busy");
      try {
        landmarker = await loadEngine();
      } catch {
        // release() directly rather than clicking stop: the stop handler writes its own
        // copy, which would replace this error before the visitor could read it.
        release();
        say("Couldn't load the tracking engine. Check your connection and try again — nothing was sent anywhere.", "error");
        return;
      }

      modeButtons.forEach((button) => { if (button) button.disabled = false; });
      setMode(mode);
      stopped = false;
      lastVideoTime = -1;
      say("<b>Tracking on your device.</b> Move your head, blink, smile — the mesh follows.");
      raf = requestAnimationFrame(loop);
    });

    // Don't burn battery animating a tab nobody is looking at.
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) {
        cancelAnimationFrame(raf);
        raf = 0;
      } else if (!stopped && landmarker) {
        raf = requestAnimationFrame(loop);
      }
    });

    addEventListener("pagehide", () => stream?.getTracks().forEach((track) => track.stop()));
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
