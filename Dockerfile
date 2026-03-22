FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    libosmesa6-dev \
    freeglut3-dev \
    libgles2-mesa-dev \
    libglib2.0-0 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /appWORKING

COPY environment.yml .
RUN conda env create -f environment.yml

COPY . .

ENV LIBGL_ALWAYS_SOFTWARE=1
ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]



