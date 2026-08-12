# Epsilon demo — Hugging Face Spaces (Docker SDK).
#
# The Space repo holds only this file and README.md. Everything else is pulled
# at build time: the package from GitHub, the weights from a HF model repo.
# That keeps the Space small and means a `git push` to the main repo is enough
# to ship a code change — you just restart the Space.
#
# CPU-only wheels on purpose: the free Spaces tier has no GPU, and the CUDA
# build of torch is ~2.5 GB of wheels that would never be used. If you move
# this to ZeroGPU, drop the --index-url line so pip resolves the CUDA build.

FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

# Spaces runs the container as uid 1000; installing as root then running as a
# different user is the classic way to end up with an unwritable cache.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface
WORKDIR $HOME/app

# torch first and alone, so the CPU index does not get applied to the rest.
RUN pip install --no-cache-dir --user \
      torch --index-url https://download.pytorch.org/whl/cpu

# The app itself, straight from the repo. huggingface_hub is listed explicitly
# because start.sh needs it to pull the weights, and it lives in the [data]
# extra alongside pyarrow — which this image has no use for.
ARG EPSILON_REF=main
RUN pip install --no-cache-dir --user \
      "epsilon[web] @ git+https://github.com/aaarvs07ranger/epsilon@${EPSILON_REF}" \
      "huggingface_hub>=0.25"

COPY --chown=user start.sh .
RUN chmod +x start.sh

# Which weights to pull, and from where. Override in the Space's Variables tab
# without rebuilding.
ENV EPSILON_MODEL_REPO=aaravs07/epsilon-imagenet64 \
    EPSILON_MODEL_FILES=unet_60k.pt,dit_100k.pt

EXPOSE 7860
CMD ["./start.sh"]
