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

    def use_file(self, f):
        if not f.is_file(): return False 
        if not f.match('*.jpg'): return False
        if f.match('*web.*') or f.match('*end grain*'): return False
        return True

    def images(self):
        images = []
        for x in self.image_files:
            dataset = 'valid' if random.random() < 0.2 else 'train'
            images.append(WoodPicture(self.name, x, dataset))
        return images

print("Finding all images...")
data_root=Path('/data/wood-species/hand-cleaned')
data_dirs = [x for x in data_root.iterdir() if x.is_dir()]
woods = [WoodPictures(x) for x in data_dirs]
images = [x for wood in woods for x in wood.images()]

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
    splitter=FuncSplitter(lambda o: o.dataset == 'valid'),,
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms(),
).dataloaders(images, bs=32)

# data_dir = Path("/data/")
# latest_dump = str(sorted(data_dir.glob("wood-identifier-*.pkl.pth"))[-1]).replace('.pth', '')
# print(f"Loading the latest version of the model from '{latest_dump}'...")

learn = vision_learner(dls, resnet50, metrics=error_rate)
# learn.load(latest_dump)

lr_mult = 100
pct_start = 0.3
div = 5.0
base_lr = 1e-05

print("Running the first 'freeze' epoch...")
learn.freeze()
learn.fit_one_cycle(1, slice(base_lr), pct_start=0.99)
base_lr /= 2
learn.unfreeze()

for epoch in range(30):
    print("Running epoch: ", epoch)
    learn.fit_one_cycle(1, slice(base_lr/lr_mult, base_lr), pct_start = pct_start, div = div)
    ts = int(time.time())
    save_file = f"/data/wood-identifier-{ts}"
    print(f"Saving the model to '{save_file}'...")
    learn.save(save_file)
    print("Epoch complete!")
    print("---------------------------------------------------")

print("Current state of data...")
os.system('ls -la /data')

