from torch.utils.data import Dataset as _TorchDataset

from ..config import CfgNode


class COCODataset(_TorchDataset):

    def __init__(self, images, annotations_by_image, transforms=None):
        self.images = images
        self.annotations_by_image = annotations_by_image
        self.transforms = transforms

    @classmethod
    def from_config(cls, cfg: CfgNode):
        images = []
        annotations_by_image = []

        # TODO

        return cls(images, annotations_by_image)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        im = self.images[i]
        ann = self.annotations_by_image[i]
        if self.transforms:
            im = self.transforms(im)
        return dict(image=im, target=ann)
