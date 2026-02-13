import torch


def string2tensor(s: str) -> torch.Tensor:
    # string -> UTF-8 bytes -> torch.Tensor
    return torch.ByteTensor(list(s.encode('utf-8')))


def tensor2string(tensor: torch.Tensor) -> str:
    # torch.Tensor -> UTF-8 bytes -> string
    bytes_data = bytes(tensor.tolist())
    return bytes_data.decode('utf-8').strip('\x00')
