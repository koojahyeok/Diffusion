import torch.utils.data


def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)        # consts tensor에서 마지막 차원을 기준으로 't' 텐서에 해당하는 부분을 가져와 c에 할당
    return c.reshape(-1, 1, 1, 1)   # c 텐서를 재구성하여 반환, -1은 해당 차원의 크기를 자동으로 결정하라는 의미, '1,1,1'은 3D텐서로 변형하기 위해 차원 추가