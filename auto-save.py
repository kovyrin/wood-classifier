#!/usr/bin/env python

import time
from pathlib import Path
from fastai.vision.all import *

class WoodPicture:
    def __init__(self, name, path, dataset):
        self.name = name
        self.path = path
        self.dataset = dataset

    def image(self):
        return Image.open(self.path)

class WoodPictures:
    def __init__(self, path):
        self.path = path
        self.name = path.name
        self.image_files = [x for x in path.iterdir() if self.use_file(x)]
        self.images = None

    def use_file(self, f):
        if not f.is_file(): return False 
        if not f.match('*.jpg'): return False
        if f.match('*web.*') or f.match('*end grain*'): return False
        return True

    def get_images(self):
        if self.images is None:
            self.images = []
            have_train = False
            for x in self.image_files:
                dataset = 'valid' if random.random() < 0.2 else 'train'
                if dataset == 'train': have_train = True
                self.images.append(WoodPicture(self.name, x, dataset))
            if not have_train: self.images[0].dataset = 'train'
        return self.images

class CustomSaveCallback(SaveModelCallback):
    def _save(self, name):
        best_fname = f'{name}_{self.best}'
        self.last_saved_path = self.learn.save(best_fname, with_opt=self.with_opt)
        print("Saved to ", self.last_saved_path)

print("Finding all images...")
data_root=Path('/data/wood-species/hand-cleaned')
data_dirs = [x for x in data_root.iterdir() if x.is_dir()]
woods = [WoodPictures(x) for x in data_dirs]
woods = [wood for wood in woods if len(wood.image_files) > 5]
images = [x for wood in woods for x in wood.get_images()]

def get_images(x):
    return x

def get_path(x):
    return x.path

def get_name(x):
    return x.name

dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_images,
    get_x=get_path,
    get_y=get_name,
    splitter=FuncSplitter(lambda o: o.dataset == 'valid'),
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=[*aug_transforms(), RandomErasing(p=0.5, max_count=10, sh=0.1)],
).dataloaders(images, bs=32)

model_dir = Path("/data/auto-save")
model_dir.mkdir(parents=True, exist_ok=True)

learn = vision_learner(dls, resnet50, metrics=error_rate, path=model_dir)
autoload_data = Path("/data/auto-save/models")

best_model = str(sorted(autoload_data.glob("model_0.*.pth"))[0]).replace('.pth', '')
print("Using the best model: ", best_model)
learn.load(best_model)

while True:
    lr = learn.lr_find()
    lr_max = lr.valley

    print("Training with lr_max=", lr_max)
    learn.fit_one_cycle(
        20, 
        lr_max=slice(lr_max), 
        cbs=[
            CustomSaveCallback(with_opt=True), 
            EarlyStoppingCallback(monitor='valid_loss', min_delta=0.01, patience=5)
        ]
    )

