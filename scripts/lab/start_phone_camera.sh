#!/bin/bash
# =============================================================================
# Phone camera bridge: DroidCam (Android, USB) -> ffmpeg -> /dev/video10
# =============================================================================
#
# Keeps running and reconnects automatically if the phone stream drops.
# Leave it running in its own terminal / tmux window. Ctrl+C to stop.
#
# Before running: phone plugged into the Pi by USB, USB debugging on,
# DroidCam app open on its main screen.
#
# Environment variables:
#   DROIDCAM_PORT  - Port shown in the DroidCam app (default: 4747)
#   PHONECAM_DEV   - v4l2loopback device (default: /dev/video10)
#   CAM_WIDTH / CAM_HEIGHT / CAM_FPS - Output format (default: 800x600 @ 30)
# =============================================================================

DROIDCAM_PORT="${DROIDCAM_PORT:-4747}"
PHONECAM_DEV="${PHONECAM_DEV:-/dev/video10}"
CAM_WIDTH="${CAM_WIDTH:-800}"
CAM_HEIGHT="${CAM_HEIGHT:-600}"
CAM_FPS="${CAM_FPS:-30}"
STREAM_URL="http://127.0.0.1:${DROIDCAM_PORT}/video"

trap 'echo ""; echo "Phone camera bridge stopped."; exit 0' SIGINT SIGTERM

# Virtual camera device
if [ ! -e "$PHONECAM_DEV" ]; then
    echo "$PHONECAM_DEV missing -- loading v4l2loopback (needs sudo)..."
    sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="PhoneCam" exclusive_caps=1
fi

while true; do
    # 1. Phone connected and authorized?
    state="$(adb get-state 2>/dev/null)"
    if [ "$state" != "device" ]; then
        echo "[phonecam] Waiting for phone (adb state: ${state:-none}). Plug in USB, tap 'Allow' on the phone."
        sleep 3
        continue
    fi

    # 2. Forward the DroidCam port over USB (re-done every loop; it drops on reconnect)
    adb forward "tcp:${DROIDCAM_PORT}" "tcp:${DROIDCAM_PORT}" >/dev/null

    # 3. DroidCam serving video? (the stream never ends, so curl times out; check the HTTP code)
    http_code="$(curl -s -o /dev/null --max-time 2 -w '%{http_code}' "$STREAM_URL")"
    if [ "$http_code" != "200" ]; then
        echo "[phonecam] Phone connected but DroidCam is not serving (HTTP ${http_code:-none}). Open the DroidCam app (main screen)."
        sleep 3
        continue
    fi
    sleep 1  # DroidCam serves one client at a time; let it release the check connection

    # 4. Bridge into the virtual camera (blocks until the stream ends or stalls)
    echo "[phonecam] Streaming $STREAM_URL -> $PHONECAM_DEV (${CAM_WIDTH}x${CAM_HEIGHT} @ ${CAM_FPS} fps)"
    ffmpeg -nostdin -loglevel error -fflags nobuffer -flags low_delay \
        -rw_timeout 5000000 \
        -i "$STREAM_URL" \
        -vf "scale=${CAM_WIDTH}:${CAM_HEIGHT},format=yuyv422" -r "$CAM_FPS" \
        -f v4l2 "$PHONECAM_DEV"
    echo "[phonecam] Stream ended (exit $?). Reconnecting in 2s..."
    sleep 2
done
