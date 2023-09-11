import torch
from infrastructure.handlers.track import Tracker


if __name__ == '__main__':
    tracker = Tracker(config_path='../settings/config.yml')
    
    with torch.no_grad():
        tracker.detect()