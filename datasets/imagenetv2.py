from torchvision import datasets
from .imagenet import ImageNet
from .imagenet import imagenet_classes
import os
import torchvision.transforms as transforms

imagenet_templates = ["itap of a {}.",
                      "a bad photo of the {}.",
                      "a origami {}.",
                      "a photo of the large {}.",
                      "a {} in a video game.",
                      "art of the {}.",
                      "a photo of the small {}."]

class ImageNetV2():
    """ImageNetV2.

    This dataset is used for testing only.
    """

    dataset_dir = "imagenetv2"

    # def __init__(self):
    #     # root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
    #     root = '../Tip-Adapter/data/'
    #     self.dataset_dir = os.path.join(root, self.dataset_dir)
    #     image_dir = "imagenetv2-matched-frequency-format-val"
    #     self.image_dir = os.path.join(self.dataset_dir, image_dir)
    #
    #     text_file = os.path.join(self.dataset_dir, "classnames.txt")
    #     classnames = ImageNet.read_classnames(text_file)
    #
    #     data = self.read_data(classnames)
    #
    #     super().__init__(train_x=data, test=data)

    def __init__(self, root, preprocess):

        self.dataset_dir = os.path.join(root, "imagenetv2")
        self.image_dir = os.path.join(self.dataset_dir, "imagenetv2-matched-frequency-format-val")

        train_preprocess = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        test_preprocess = preprocess

        self.dataset = datasets.ImageFolder(root=self.image_dir, transform=test_preprocess)
        class_to_idx = self.dataset.class_to_idx
        inv_class_to_idx = {v: int(k) for k, v in class_to_idx.items()}
        self.dataset.idx_to_idx = inv_class_to_idx

        self.template = imagenet_templates
        self.classnames = imagenet_classes

    # def read_data(self, classnames):
    #     image_dir = self.image_dir
    #     folders = list(classnames.keys())
    #     items = []
    #
    #     for label in range(1000):
    #         class_dir = os.path.join(image_dir, str(label))
    #         imnames = listdir_nohidden(class_dir)
    #         folder = folders[label]
    #         classname = classnames[folder]
    #         for imname in imnames:
    #             impath = os.path.join(class_dir, imname)
    #             item = Datum(impath=impath, label=label, classname=classname)
    #             items.append(item)
    #
    #     return items