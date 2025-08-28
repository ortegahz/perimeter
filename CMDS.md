# play
ffplay -fflags nobuffer -flags low_delay -protocol_whitelist file,udp,rtp /home/manu/tmp/track.sdp

ffplay -fflags nobuffer -flags low_delay \
       -probesize 32 -analyzeduration 0 \
       -framedrop -sync ext \
       udp://127.0.0.1:5000

ffplay -fflags nobuffer -flags low_delay udp://127.0.0.1:5000
ffplay -fflags nobuffer -flags low_delay udp://127.0.0.1:5001

# insightface
pip install nvidia-cudnn-cu12==9.0.0.312 
export LD_LIBRARY_PATH=/home/manu/anaconda3/envs/yolo/lib/python3.8/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# onnx
onnxsim /home/manu/.insightface/models/buffalo_l/det_10g.onnx /home/manu/.insightface/models/buffalo_l/det_10g_simplified.onnx
onnxsim /home/manu/.insightface/models/buffalo_l/w600k_r50.onnx /home/manu/.insightface/models/buffalo_l/w600k_r50_simplified.onnx
