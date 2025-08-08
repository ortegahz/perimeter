# play

ffplay -fflags nobuffer -flags low_delay -protocol_whitelist file,udp,rtp /home/manu/tmp/track.sdp

ffplay -fflags nobuffer -flags low_delay \
       -probesize 32 -analyzeduration 0 \
       -framedrop -sync ext \
       udp://127.0.0.1:5000

ffplay -fflags nobuffer -flags low_delay udp://127.0.0.1:5000

# insightface

export LD_LIBRARY_PATH=/home/manu/anaconda3/envs/yolo/lib/python3.8/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
