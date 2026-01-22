import yaml
import os
import sys

class Config:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = os.path.dirname(os.path.dirname(current_dir)) # EyesTracking/

        settings_path = os.path.join(current_dir, 'settings.yaml')
        with open(settings_path, 'r', encoding='utf-8') as f:
            self.settings = yaml.safe_load(f)

        paths_path = os.path.join(current_dir, 'paths.yaml')
        with open(paths_path, 'r', encoding='utf-8') as f:
            self.paths = yaml.safe_load(f)

    def get_data_dir(self):
        """Trả về đường dẫn tuyệt đối đến thư mục data"""
        return os.path.join(self.root_dir, self.paths['data']['base_dir'])

    def get_output_csv_path(self):
        """Trả về đường dẫn tuyệt đối đến file CSV đầu ra"""
        return os.path.join(self.get_data_dir(), self.paths['data']['output_filename'])

    def get_models_dir(self):
        """Trả về đường dẫn tuyệt đối đến thư mục models"""
        return os.path.join(self.root_dir, self.paths['models']['base_dir'])

    def get_model_path(self, model_name=None):
        """Trả về đường dẫn tuyệt đối đến file model"""
        if model_name is None:
            model_name = self.paths['models']['default_model']
        return os.path.join(self.get_models_dir(), model_name)

    def get_scaler_path(self):
        """Trả về đường dẫn tuyệt đối đến file scaler"""
        return os.path.join(self.get_models_dir(), self.paths['models']['scaler'])

cfg = Config()