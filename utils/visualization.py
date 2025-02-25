from dataset import FULL_BODY_KEYS, KINEMATIC_TREE


# 3Dポーズの描画
def draw_3d_pose(ax, keypoints3d, color='green', linecolor='black'):
    
    # キーポイントを描画
    for x, y, z in keypoints3d:
        ax.scatter(x, z, y, c=color, marker='o')

    # 接続線を描画
    for chain in KINEMATIC_TREE:
        for j in range(len(chain)-1):
            start_idx, end_idx = FULL_BODY_KEYS.index(chain[j]), FULL_BODY_KEYS.index(chain[j+1])
            
            if start_idx is not None and end_idx is not None:
                start_point = keypoints3d[start_idx]
                end_point = keypoints3d[end_idx]
                ax.plot(
                    [start_point[0], end_point[0]],  # x座標
                    [start_point[2], end_point[2]],  # z座標
                    [start_point[1], end_point[1]],  # y座標
                    c=linecolor  # 線の色
                )
    return ax