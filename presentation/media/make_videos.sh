#!/bin/bash
# Builds the four real videos of the talk (presentation/media/*.mp4) from the trial videos in results/ (the results
# store on the Hub) and a raw dataset episode from the Hub. Re-run after changing a label; then rebuild the deck.
set -e
cd ~/Projects/drtc-major
E=results/evals/cubebox_v1_drtc_v2/sessions
P=$(ls -d $E/*/trials/upstream_patient__mix/grootn17_cubebox_v1_full/t01)/video
D=$(ls -d $E/*/trials/drtc__mix/grootn17_cubebox_v1_full/t01)/video
K=$(ls -d $E/*/trials/drtc_stack_nortc__mix/grootn17_cubebox_v1_full/t01)/video
B=$(pwd)/presentation/fonts/Inter-SemiBold.ttf
R=$(pwd)/presentation/fonts/Inter-Regular.ttf
O=presentation/media
ENC="-c:v libx264 -preset slow -crf 26 -pix_fmt yuv420p -movflags +faststart -an"
tag() { echo "drawtext=fontfile=$B:text='$1':x=$2:y=$3:fontsize=$4:fontcolor=white:box=1:boxcolor=$5:boxborderw=14$6"; }

# 1. hook: the patient run of GR00T N1.7 trial 1; the faults hit and the arm stops
ffmpeg -v error -y -i $P/scene.mp4 -vf "scale=960:720,$(tag 'network faults' 24 24 30 0xEB6834@0.92 ":enable='between(t,4.8,10.3)'"),$(tag 'no new actions - the arm waits' 24 660 30 0x15191F@0.85 ":enable='gte(t,7.4)'")" $ENC $O/hook.mp4

# 2. record: a raw demonstration from the dataset on the Hub (cube in box v1, episode 0, 9.6 s)
# the episode's videos, downloaded once from the Hub into the deck's cache (needs read access to the dataset)
HUB=presentation/.cache/hub/so101_cubebox_v1_raw
for c in camera1 camera2; do
  [ -f $HUB/videos/observation.images.$c/chunk-000/file-000.mp4 ] || .venv/bin/python -c "from huggingface_hub import hf_hub_download; hf_hub_download('orangemoon599/so101_cubebox_v1_raw', 'videos/observation.images.$c/chunk-000/file-000.mp4', repo_type='dataset', local_dir='$HUB')"
done
C1=$HUB/videos/observation.images.camera1/chunk-000/file-000.mp4
C2=$HUB/videos/observation.images.camera2/chunk-000/file-000.mp4
ffmpeg -v error -y -t 9.57 -i $C2 -t 9.57 -i $C1 -filter_complex "[0:v]scale=800:600,$(tag 'scene camera' 16 548 24 0x15191F@0.85 '')[a];[1:v]scale=800:600,$(tag 'wrist camera' 16 548 24 0x15191F@0.85 '')[b];[a][b]hstack=inputs=2" $ENC $O/record.mp4

# 3. robot_eval: a live trial, the DRTC run of the same trial, both cameras
ffmpeg -v error -y -i $D/scene.mp4 -i $D/wrist.mp4 -filter_complex "[0:v]scale=800:600,$(tag 'scene camera' 16 548 24 0x15191F@0.85 '')[a];[1:v]scale=800:600,$(tag 'wrist camera' 16 548 24 0x15191F@0.85 '')[b];[a][b]hstack=inputs=2" $ENC $O/robot_eval.mp4

# 4. e4_trials: one more trial (Diffusion Policy, trial 10: same faults for all three) under three clients, side by side;
#    each holds its last frame. Patient: ended by the client at 14.3 s; DRTC done at 13.0 s; the stack at 12.3 s.
P4=$(ls -d $E/*/trials/upstream_patient__mix/diffusion_cubebox_v1_full/t10)/video
D4=$(ls -d $E/*/trials/drtc__mix/diffusion_cubebox_v1_full/t10)/video
K4=$(ls -d $E/*/trials/drtc_stack_nortc__mix/diffusion_cubebox_v1_full/t10)/video
L=14.3
ffmpeg -v error -y -i $P4/scene.mp4 -i $D4/scene.mp4 -i $K4/scene.mp4 -filter_complex "\
[0:v]scale=640:480,tpad=stop_mode=clone:stop_duration=3,$(tag 'Patient LeRobot' 14 14 24 0x2A78D6@0.92 ''),$(tag 'run ended' 14 420 24 0xEB6834@0.95 ":enable='gte(t,14.1)'")[a];\
[1:v]scale=640:480,tpad=stop_mode=clone:stop_duration=3,$(tag 'DRTC' 14 14 24 0x1BAF7A@0.92 ''),$(tag 'task done' 14 420 24 0x1BAF7A@0.95 ":enable='gte(t,12.9)'")[b];\
[2:v]scale=640:480,tpad=stop_mode=clone:stop_duration=3,$(tag 'DRTC + stack' 14 14 24 0xB07D00@0.92 ''),$(tag 'task done' 14 420 24 0x1BAF7A@0.95 ":enable='gte(t,12.2)'")[c];\
[a][b][c]hstack=inputs=3,trim=0:$L" $ENC $O/e4_trials.mp4
ls -la $O/*.mp4
