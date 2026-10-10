/**
 * Proves the browser boost and the Python boost are the same algorithm.
 *
 * `tests/test_boost_parity.py` generates, from the Python implementation, the four pass
 * buffers it used, the canvas it produced from them, and the source rectangles it sampled —
 * then runs this file. Two things are asserted:
 *
 *   1. `sourceRectForPhase()` returns the rectangles Python actually sampled. This is the
 *      geometry, and it is where the two could silently drift: a sign error here looks like a
 *      sharpening filter rather than a bug.
 *   2. `interleave()` reproduces Python's canvas byte for byte from the same four passes.
 *
 * What is not compared is the interpolation, because there isn't one to compare: Python uses
 * `cv2.warpAffine` with cubic interpolation, the browser uses `drawImage`. Same geometry,
 * different filter, and pretending otherwise would make this test lie.
 */
import { readFile } from "node:fs/promises";
import { copyFile, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const fixtureDirectory = process.argv[2];
if (!fixtureDirectory) {
  console.error("usage: node boost_parity.mjs <fixture-directory>");
  process.exit(2);
}

// The browser file is an ES module served as .js; node needs the extension to match the
// parse goal, so it is imported from a copy rather than through a package.json.
const staging = await mkdtemp(path.join(tmpdir(), "boost-parity-"));
const moduleCopy = path.join(staging, "boost.mjs");
await copyFile(path.join(here, "..", "static", "boost.js"), moduleCopy);

let boost;
try {
  boost = await import(pathToFileURL(moduleCopy).href);
} finally {
  await rm(staging, { recursive: true, force: true });
}

const [rects, passes, expected] = await Promise.all([
  readFile(path.join(fixtureDirectory, "rects.json"), "utf8").then(JSON.parse),
  readFile(path.join(fixtureDirectory, "passes.bin")),
  readFile(path.join(fixtureDirectory, "expected.bin")),
]);

const failures = [];
const scale = rects.scale;

// ── 1. the geometry ──
boost.phases(scale).forEach(([phaseX, phaseY], index) => {
  const rect = boost.sourceRectForPhase(rects.region, scale, phaseX, phaseY);
  const want = rects.rects[index];
  for (const key of ["x", "y", "w", "h"]) {
    if (Math.abs(rect[key] - want[key]) > 1e-9) {
      failures.push(
        `phase (${phaseX}, ${phaseY}): ${key} is ${rect[key]}, Python sampled ${want[key]}`,
      );
    }
  }
});

// ── 2. the interleave ──
const passLength = boost.ALIGN * boost.ALIGN * 3;
const collected = boost.phases(scale).map(([x, y], index) => ({
  x,
  y,
  width: boost.ALIGN,
  height: boost.ALIGN,
  channels: 3,
  data: passes.subarray(index * passLength, (index + 1) * passLength),
}));

let canvas;
try {
  canvas = boost.interleave(collected, scale, 3);
} catch (error) {
  console.error(`PARITY FAILED: interleave threw: ${error.message}`);
  process.exit(1);
}

const size = boost.ALIGN * scale;
if (canvas.width !== size || canvas.height !== size) {
  failures.push(`canvas is ${canvas.width}x${canvas.height}, expected ${size}x${size}`);
} else if (canvas.data.length !== expected.length) {
  failures.push(`canvas holds ${canvas.data.length} bytes, Python produced ${expected.length}`);
} else {
  let differing = 0;
  let worst = 0;
  let firstAt = -1;
  for (let index = 0; index < expected.length; index += 1) {
    const delta = Math.abs(canvas.data[index] - expected[index]);
    if (delta !== 0) {
      differing += 1;
      if (firstAt < 0) firstAt = index;
      worst = Math.max(worst, delta);
    }
  }
  if (differing) {
    const pixel = Math.floor(firstAt / 3);
    failures.push(
      `${differing} of ${expected.length} bytes differ, worst by ${worst}, ` +
        `first at canvas (${Math.floor(pixel / size)}, ${pixel % size})`,
    );
  }
}

if (failures.length) {
  console.error("PARITY FAILED");
  for (const failure of failures) console.error(`  - ${failure}`);
  process.exit(1);
}

console.log(
  `PARITY OK: ${scale}x, ${collected.length} phases, ${size}x${size} canvas, ` +
    `${expected.length} bytes identical`,
);
