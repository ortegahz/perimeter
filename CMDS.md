# play

ffplay -fflags nobuffer -flags low_delay -protocol_whitelist file,udp,rtp /home/manu/tmp/track.sdp

ffplay -fflags nobuffer -flags low_delay \
       -probesize 32 -analyzeduration 0 \
       -framedrop -sync ext \
       udp://127.0.0.1:5000