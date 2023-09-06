
class CoCoRDTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        im_sq = self.base_transform(x)
        im_sk = self.base_transform(x)
        im_tk = self.base_transform(x)
        return [im_sq, im_sk, im_tk]


