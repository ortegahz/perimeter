# perimeter
perimeter security algorithm

# dla
/usr/src/tensorrt/bin/trtexec --onnx=/mnt/nfs/w600k_r50_simplified.onnx --saveEngine=w600k_r50_simplified.dla.engine --useDLACore=0 --allowGPUFallback --verbose
/usr/src/tensorrt/bin/trtexec --onnx=/mnt/nfs/model.onnx --saveEngine=/mnt/nfs/w600k_r50_simplified.dla.engine --useDLACore=0 --allowGPUFallback --verbose

/usr/src/tensorrt/bin/trtexec --onnx=/mnt/nfs/det_10g_simplified.onnx --saveEngine=/mnt/nfs/det_10g_simplified.dla.engine --useDLACore=0 --allowGPUFallback --verbose
/usr/src/tensorrt/bin/trtexec \
    --onnx=/mnt/nfs/det_10g_simplified.onnx \
    --saveEngine=/mnt/nfs/det_10g_simplified.gpu.engine \
    --int8 \
    --calib=/mnt/nfs/det_10g.calib \
    --verbose
/usr/src/tensorrt/bin/trtexec \
    --onnx=/mnt/nfs/det_10g_simplified.onnx \
    --saveEngine=/mnt/nfs/det_10g_simplified.gpu.engine \
    --int8 \
    --calib=/mnt/nfs/det_10g.calib \
    --loadInputs=/mnt/nfs/calibration_files.txt \
    --verbose
/usr/src/tensorrt/bin/trtexec \
    --onnx=/mnt/nfs/det_10g_simplified.onnx \
    --saveEngine=/mnt/nfs/det_10g_simplified.dla.engine \
    --useDLACore=0 \
    --allowGPUFallback \
    --int8 \
    --calib=/mnt/nfs/det_10g.calib \
    --loadInputs=/mnt/nfs/calibration_files.txt \
    --verbose
/usr/src/tensorrt/bin/trtexec \
    --onnx=/mnt/nfs/det_10g_simplified.onnx \
    --saveEngine=/mnt/nfs/det_10g_simplified.dla.engine \
    --useDLACore=0 \
    --allowGPUFallback \
    --int8 \
    --fp16 \
    --best \
    --calib=/mnt/nfs/det_10g.calib \
    --loadInputs=/mnt/nfs/calibration_files.txt \
    --memPoolSize=workspace:6700M \
    --verbose

/usr/src/tensorrt/bin/trtexec --onnx=/mnt/nfs/reid_model.onnx --saveEngine=/home/nvidia/VSCodeProject/smartboxcore/models/tensorrt/reid_model_dla.engine --useDLACore=1 --allowGPUFallback --verbose

/usr/src/tensorrt/bin/trtexec --onnx=/mnt/nfs/Resnet50_epoch_0.onnx --saveEngine=/mnt/nfs/Resnet50_epoch_0.dla.engine --useDLACore=0 --allowGPUFallback
/usr/src/tensorrt/bin/trtexec --onnx=/mnt/nfs/mobilenet0.25_epoch_0.onnx --saveEngine=/mnt/nfs/mobilenet0.25_epoch_0.dla.engine --useDLACore=0 --allowGPUFallback
/usr/src/tensorrt/bin/trtexec --onnx=/mnt/nfs/retinaface_pytorch_mn025_relu/mobilenet0.25_Final.onnx --saveEngine=/mnt/nfs/mobilenet0.25_Final.dla.engine --useDLACore=0 --allowGPUFallback