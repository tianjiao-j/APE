from torchvision import datasets
from .imagenet import ImageNet
import os
import torchvision.transforms as transforms
from .imagenet import imagenet_classes


imagenet_templates = ["itap of a {}.",
                        "a bad photo of the {}.",
                        "a origami {}.",
                        "a photo of the large {}.",
                        "a {} in a video game.",
                        "art of the {}.",
                        "a photo of the small {}."]

# @DATASET_REGISTRY.register()
class ImageNetSketch():
    """ImageNet-Sketch.

    This dataset is used for testing only.
    """

    dataset_dir = "imagenet-sketch"

    # def __init__(self, cfg):
    #     root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
    #     self.dataset_dir = os.path.join(root, self.dataset_dir)
    #     self.image_dir = os.path.join(self.dataset_dir, "images")
    #
    #     text_file = os.path.join(self.dataset_dir, "classnames.txt")
    #     classnames = ImageNet.read_classnames(text_file)
    #
    #     data = self.read_data(classnames)
    #
    #     super().__init__(train_x=data, test=data)

    def __init__(self, root, preprocess):

        self.dataset_dir = os.path.join(root, "imagenet-sketch")
        self.image_dir = os.path.join(self.dataset_dir, "images")

        train_preprocess = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        test_preprocess = preprocess

        self.dataset = datasets.ImageFolder(root=self.image_dir, transform=test_preprocess)

        self.template = imagenet_templates

        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        indices, folders, classnames = ImageNet.read_classnames(text_file)
        self.classnames = imagenet_classes
        class_to_idx = {folder: index for folder, index in zip(folders, indices)}
        inv_class_to_idx = {v: k for k, v in self.dataset.class_to_idx.items()}
        self.dataset.idx_to_idx = {i: class_to_idx[inv_class_to_idx[i]] for i in self.dataset.targets}

    # def read_data(self, classnames):
    #     image_dir = self.image_dir
    #     folders = listdir_nohidden(image_dir, sort=True)
    #     items = []
    #
    #     for label, folder in enumerate(folders):
    #         imnames = listdir_nohidden(os.path.join(image_dir, folder))
    #         classname = classnames[folder]
    #         for imname in imnames:
    #             impath = os.path.join(image_dir, folder, imname)
    #             item = Datum(impath=impath, label=label, classname=classname)
    #             items.append(item)
    #
    #     return items