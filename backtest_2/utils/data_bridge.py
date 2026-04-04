import pandas as pd
import numpy as np
import os

class Handler(dict):
    """
    檔案型 Handler：
    - key 對應到 {path}/{key}.{data_type} 檔案
    - 支援 parquet / pkl
    - 可設定 reindex_like 在讀取時自動對齊 index/columns
    """
    def __init__(self, path, data_type: str = 'parquet'):
        self.path = path
        self.cashe_dict = {'bool': bool}
        self.data_cache = {}
        self.func_dict = {}

        if data_type == 'pickle':
            data_type = 'pkl'
        self.data_type = data_type
        self.reindex_like = None
        self.start_date = None
        self.end_date = None
        os.makedirs(path, exist_ok=True)

    def _apply_view(self, df):
        if self.reindex_like is not None:
            return df.reindex_like(self.reindex_like)
        start = self.start_date
        end = self.end_date
        if start is not None or end is not None:
            return df.loc[start:end]
        return df

    def __getitem__(self, key):
        if key in self.cashe_dict:
            return self.cashe_dict[key]
        elif key in self.func_dict:
            return self.func_dict[key]
        elif key in self.data_cache:
            return self._apply_view(self.data_cache[key])
        else:
            file_path = os.path.join(self.path, f'{key}.{self.data_type}')
            if os.path.exists(file_path):
                try:
                    if self.data_type == 'parquet':
                        df = pd.read_parquet(file_path)
                    elif self.data_type == 'pkl':
                        df = pd.read_pickle(file_path)
                    else:
                        raise ValueError(f"不支援的 data_type: {self.data_type}")

                    self.data_cache[key] = df
                    return self._apply_view(df)
                except Exception:
                    raise KeyError(f'文件 {file_path} 損壞或無法讀取')
            raise KeyError(key)

    def __call__(self, key):
        return self.__getitem__(key)

    def __setitem__(self, key, value):
        """
        存檔：只對 value 中 isfinite 的值存 parquet/pkl。
        """
        file_path = os.path.join(self.path, f'{key}.{self.data_type}')
        cached_value = value[np.isfinite(value)]
        if self.data_type == 'parquet':
            cached_value.to_parquet(file_path)
        elif self.data_type == 'pkl':
            cached_value.to_pickle(file_path)
        else:
            raise ValueError(f"不支援的 data_type: {self.data_type}")
        self.data_cache[key] = cached_value

    def cash_list(self):
        """
        回傳目前資料夾底下所有對應 data_type 的 key 清單（不含副檔名）。
        """
        file_set = set(filter(lambda x: x.endswith(f".{self.data_type}"), os.listdir(self.path)))
        return sorted(map(lambda x: x[:-(len(self.data_type) + 1)], list(file_set)))