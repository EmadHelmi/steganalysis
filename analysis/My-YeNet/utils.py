import os
import random

from torch.utils.data.dataset import Dataset


class ToTensor(object):
    def __call__(self, samples):
        images, labels = samples['images'], samples['labels']
        images = images.transpose((0, 3, 1, 2))
        # images = (images.transpose((0,3,1,2)).astype('float32') / 127.5) - 1.
        return {'images': torch.from_numpy(images),
                'labels': torch.from_numpy(labels).long()}


class Resize(object):
    def __call__(self, samples):
        pass


class RandomRotate(object):
    def __call__(self, samples):
        pass


class RandomFlip(object):
    def __call__(self, samples):
        pass


class MakeDataset(Dataset):
    def __init__(self, dspath, is_cover, transform=None):
        self.data = [os.path.join(dspath, f)
                     for f in os.listdir(dspath)]
        self.is_cover = is_cover

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = numpy.asarray(Image.open(self.data[index]))
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = 0 if self.is_cover else 0

        return image, label


class GenerateDatasets():
    def __init__(self, boss_path, bows_path):
        super().__init__()
        self.boss_path = boss_path
        self.bows_path = bows_path

    def create_ds(self, test_to_train_ratio=0.5, validation_to_train_ratio=0.2, ds_type=1, embed_ratio=0.8):
        if ds_type == 1:
            self.cover_images = os.listdir(
                os.path.join(self.boss_path, "total", "imresize"))
            self.stego_images = os.listdir(
                os.path.join(self.boss_path, "stego", str(embed_ratio), "imresize"))

            self.train_images, self.test_images = self.split_dataset(
                os.listdir(os.path.join(self.boss_path, "imresize")),
                test_to_train_ratio
            )
            self.validation_images, self.train_images = self.split_dataset(
                self.train_images,
                validation_to_train_ratio

            )
            return self.train_images, self.test_images, self.validation_images

    def split_dataset(self, ds, ratio):
        random.shuffle(ds)
        index = len(ds) * ratio
        return ds[:index], ds[index:]
