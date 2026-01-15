
import torch

def encode_task(task_name):
    """
    Symbolic One-Hot Encoding for Tasks.
    Real VLA would use a text encoder like T5 or CLIP-Text.
    """
    task_map = {
        "PICK_DROP": [1.0, 0.0],
        "IDLE":      [0.0, 1.0]
    }
    vec = task_map.get(task_name, [0.0, 1.0]) # Default to IDLE
    return torch.tensor([vec])
