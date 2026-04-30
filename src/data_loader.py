# Step 1.2: GEANT4数据加载

import numpy as np
import h5py


class GEANTDataLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_single_event(self, event_id):
        """
        加载单个事件的GEANT4数据

        返回:
            hit_data: [num_hits, features]
                - features: [x, y, z, energy, time, channel_id, ...]
            label: 0（反陽子）、1（反重陽子）
        :param event_id:
        :return:
        """
        # 根据实际数据格式编写
        # 提取Si(Li)检出器hit点
        # 提取TOF计数器信息
        # 组合成hit_data
        pass