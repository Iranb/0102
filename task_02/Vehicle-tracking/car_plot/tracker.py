import torch
from infrastructure.handlers.track import Tracker


if __name__ == '__main__':
    # TODO 根据自己的环境修改config 路径
    tracker = Tracker(config_path='../settings/config.yml')
    
    with torch.no_grad():
        tracker.detect()
