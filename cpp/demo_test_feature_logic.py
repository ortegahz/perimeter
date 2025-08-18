import sys

sys.path.append("/media/manu/ST2000DM005-2U91/workspace/perimeter/cpp/cmake-build-debug")
import feature_logic

pkt1 = feature_logic.TrackInput()
pkt1.stream_id = "cam1"
pkt1.tid = 1
pkt1.fid = 100
pkt1.body_feat = [0.1, 0.2, 0.3]
pkt1.body_score = 0.88
pkt1.face_feat = [0.4, 0.5, 0.6]
pkt1.has_face = True

pkt2 = feature_logic.TrackInput()
pkt2.stream_id = "cam1"
pkt2.tid = 2
pkt2.fid = 101
pkt2.body_feat = [0.5, 0.6, 0.7]
pkt2.body_score = 0.92
pkt2.face_feat = [0.7, 0.8, 0.9]
pkt2.has_face = True

# 调用 C++ 逻辑
result = feature_logic.process_tracks([pkt1, pkt2])

print("结果1：完整返回结构字典")
print(result)

print("结果2：逐条打印")
for cam_id, tid_map in result.items():
    for tid, (gid, score, n_tid) in tid_map.items():
        print(f"cam_id={cam_id}, tid={tid}, gid={gid}, score={score}, n_tid={n_tid}")
