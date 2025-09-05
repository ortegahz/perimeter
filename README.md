# perimeter
perimeter security algorithm

# dla
/usr/src/tensorrt/bin/trtexec --onnx=/mnt/nfs/w600k_r50_simplified.onnx --saveEngine=w600k_r50_simplified.dla.engine --useDLACore=0 --allowGPUFallback --verbose
/usr/src/tensorrt/bin/trtexec --onnx=/mnt/nfs/model.onnx --saveEngine=/mnt/nfs/w600k_r50_simplified.dla.engine --useDLACore=0 --allowGPUFallback --verbose

/usr/src/tensorrt/bin/trtexec --onnx=/mnt/nfs/reid_model.onnx --saveEngine=/mnt/nfs/reid_model_dla.engine --useDLACore=1 --allowGPUFallback --verbose