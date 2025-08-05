import os

from cores.faceSearcher import FaceSearcher

# =================== 使用示例 ===================
if __name__ == "__main__":
    db_dir = "/home/manu/图片/faces/"  # 你的库目录
    query_img = "/home/manu/图片/vlcsnap-2025-08-01-18h52m04s342.png"  # 查询图片
    top_k = 5

    searcher = FaceSearcher(provider="CPUExecutionProvider")
    searcher.build_db(db_dir)  # 仅第一次建库，后面可省略
    # searcher.add("/path/to/new_face.png")        # 动态增量
    results = searcher.search(query_img, top_k=top_k)

    print("\nTop-{} 相似结果:".format(top_k))
    for rank, (img_path, score) in enumerate(results, 1):
        print(f"{rank:02d}. {os.path.basename(img_path):30s}  score={score:.4f}")
