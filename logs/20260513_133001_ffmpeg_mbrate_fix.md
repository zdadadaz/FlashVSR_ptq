# 20260513_133001 FFmpeg MB Rate Fix

## Objective
Fix "MB rate > level limit" and "More than 1000 frames duplicated" errors in `ffmpeg_comb.sh`.

## Diagnosis
- High input TBN (1000k) was causing FFmpeg to misinterpret timestamps.
- Without a forced FPS filter, `hstack` was seeing inconsistent timing between streams, leading to astronomical internal frame rate calculations.

## Changes
- Modified `ffmpeg_comb.sh`.
- Added `fps=30` filter to both input streams *inside* the filter complex to force normalization before synchronization.
- Added `-video_track_timescale 30000` to set a sane timebase for the output MP4 container.

## Execution Result
- `ffmpeg_comb.sh` now produces stable 30fps output without duplication errors.
