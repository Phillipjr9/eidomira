# Measuring the neural path on a free Kaggle GPU

For the situation this project is in: the pipeline is built, the licensed artefacts are not
bought, and no card is available to rent a GPU. Kaggle gives roughly **30 GPU-hours a week** on
a T4 (or P100), unlocked by phone verification, with **no card at any point**.

A notebook cannot host the live studio — no public IP, no inbound UDP, no persistent process.
What it can do is answer the question that has to be settled before anything is paid for: does
the neural path run on a GPU, and how fast?

There is a ready-to-import notebook: [`notebooks/kaggle-validation.ipynb`](../notebooks/kaggle-validation.ipynb).
Everything below is the same thing by hand.

## Steps

**1. Account and verification.** kaggle.com → sign up → verify a phone number in Settings.
Without verification the GPU accelerator and even Internet are unavailable, and the failure
looks like a missing option rather than a locked one.

**2. New notebook** → *File → Import Notebook* and upload the `.ipynb`, or start a blank one.

**3. Two settings, in the right-hand panel:**

| setting | value | why |
|---|---|---|
| Accelerator | **GPU T4 x2** | otherwise every number is a CPU number |
| Internet | **On** | to clone the repository and install packages |

**4. Run these four cells.**

```python
# Cell 1 — the code. The default branch is not where the work is, so name the branch.
BRANCH = "arena/178efbef-eidomira"
!nvidia-smi -L
import os
if not os.path.exists("eidomira") and os.path.basename(os.getcwd()) != "eidomira":
    !git clone -q -b {BRANCH} https://github.com/Phillipjr9/eidomira.git
if os.path.basename(os.getcwd()) != "eidomira":
    %cd eidomira
!git fetch -q origin {BRANCH} && git reset -q --hard origin/{BRANCH}
!git log --oneline -1
```

```python
# Cell 2 — a GPU build of ONNX Runtime and insightface for the swap model.
# Kaggle runs CUDA 12, so install onnxruntime-gpu==1.26.0 (CUDA 12 build) and cu12 libraries.
!pip install -q --force-reinstall onnxruntime-gpu==1.26.0
!pip install -q nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 onnx opencv-python-headless scipy scikit-image tqdm requests
!pip install -q --no-deps insightface
!python -c "import onnxruntime as o; print('Providers:', o.get_available_providers())"
```

```python
# Cell 3 — measure.
!python tools/gpu_validation.py
```

If cell 2 prints a provider list containing `CUDAExecutionProvider`, cell 3 is a GPU
measurement. If it does not, cell 3 says so and, when a GPU is attached, prints the reason ONNX
Runtime refused it.

**5. Save the output.** That is the number that decides which GPU is worth renting.

## The four ways this goes wrong

| symptom | cause | fix |
|---|---|---|
| `Repository not found`, or an old tree with no `tools/` | cloned the default branch — `main` holds the initial snapshot, not the work | add `-b arena/178efbef-eidomira` |
| GPU accelerator missing from Settings | phone not verified | verify it; the option appears afterwards |
| `will be used: CPUExecutionProvider` with a T4 attached | a CPU-only `onnxruntime` shadowing the GPU one | the uninstall in cell 2, then reinstall `onnxruntime-gpu` |
| `ImportError: libGL.so.1` | `insightface` pulls GUI `opencv-python` | `pip install --force-reinstall opencv-python-headless` |

The last two have both been hit here, which is why the cells are written the way they are.

## What the numbers prove, and what they do not

**Prove:** the plumbing runs on that host, and how fast that GPU is at these shapes. The
provider line is resolved by `app/providers.py` — the same code the studio runs, not an
inspection of what ONNX Runtime has compiled in.

**Do not prove:** quality. With the stand-ins there are no trained weights to judge. Point
`--models` at a directory holding real weights once the licence is bought and the same script
measures those; expect very different timings, because a real inswapper pass is far more work
than a shape-correct stand-in.

The frame figures are an extrapolation from a 1280×720 measurement, scaled by pixel count. That
is honest for a convolution-shaped model and a rough guide for a real one. The pixel boost runs
four swap passes, so add roughly three times the swap time at scale 2.
