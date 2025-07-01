# src/knowledge_base.py
import torch
import os
import copy

class KnowledgeBase:
    """
    服务器端的模型知识库。
    负责存储、检索和更新来自各个客户端的模型。
    """
    def __init__(self, output_dir):
        self.models = {}  # { 'Farm_A': state_dict, 'Farm_B': state_dict, ... }
        self.kb_dir = os.path.join(output_dir, 'knowledge_base')
        os.makedirs(self.kb_dir, exist_ok=True)
        print(f"知识库已初始化，模型将保存在: {self.kb_dir}")

    def update(self, farm_id: str, model_state_dict: dict):
        """用最新的模型更新或添加一个农场的模型到知识库。"""
        # 存储深拷贝以避免外部修改
        self.models[farm_id] = copy.deepcopy(model_state_dict)
        # 将其持久化到磁盘
        save_path = os.path.join(self.kb_dir, f'kb_model_{farm_id}.pth')
        torch.save(model_state_dict, save_path)
        # print(f"  [KB] 已更新 {farm_id} 的模型。")

    def get_model(self, farm_id: str):
        """获取指定农场的模型。"""
        return self.models.get(farm_id)

    def get_all_models(self):
        """获取知识库中所有模型的 state_dict 列表。"""
        return list(self.models.values())
        
    def get_all_farm_ids(self):
        """获取知识库中所有农场的ID。"""
        return list(self.models.keys())

    def __len__(self):
        return len(self.models)