#!/usr/bin/env bash
set -euo pipefail

# Placeholder script: add manual instructions & automated steps when credentials available.
# DAIC-WOZ: Requires registration. After download:
#   Place audio in data/daic_woz/audio
#   Place transcripts in data/daic_woz/transcripts
#   Extract video frames: ffmpeg -i input.mp4 -vf fps=2 data/daic_woz/frames/<id>_%04d.jpg
# IEMOCAP: Unpack sessions into data/iemocap/SessionX.
# MOSEI: Use CMU SDK or raw files into data/mosei/.

echo "Refer to README for detailed dataset acquisition steps."
