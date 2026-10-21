# The Royal Court of Vision - one container serves the website AND the API (Hugging Face Spaces: Docker SDK)
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 git && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs the container as user 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    RCV_MODELS_DIR=/home/user/app/backend/models \
    RCV_WORK_DIR=/tmp/rcv_jobs \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# small CPU-only PyTorch first, then everything else
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

# bake every model into the image (detectors, YOLO-World + CLIP with the big vocabulary, people-analysis model),
# so a visitor never waits for a download
RUN pip install --no-cache-dir "git+https://github.com/ultralytics/CLIP.git" || pip install --no-cache-dir ftfy regex
RUN cd backend && python prepare_models.py

WORKDIR /home/user/app/backend
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
