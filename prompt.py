import json

class Prompt:
    def __init__(self, json_path: str):
        self.json_path = json_path
        with open(json_path, 'r') as f:
            self.data = json.load(f)

    def get_data(self):
        return self.data
    
    def get_node_by_title(self, title: str):
        for node_id, node in self.data.items():
            if node.get('_meta', {}).get('title') == title:
                return node
        return None
    
    def set_node_field(self, title: str, value):
        node = self.get_node_by_title(title)
        if node:
            node = value
        else:
            raise ValueError(f"Node with title '{title}' not found.")