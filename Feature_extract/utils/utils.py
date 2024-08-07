
import torch
import numpy as np
import torch.nn.functional as F
import sys
sys.path.append("./utils")
import cv2
from PIL import Image
import openslide

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_features(batch):
	img = torch.stack([item[0] for item in batch], dim = 0)
	coords = np.vstack([item[1] for item in batch])
	assert len(img.shape) == 4, "img shape is wrong, please check"
	return [img, coords]

class ImgReader:
    """used for jpg, png, etc. 
    """
    default_dims = [1.0, 2.0, 4.0, 8.0, 16.0]
    def __init__(self, filename) -> None:
        self.filename = filename
        dtype = filename.split('.')[-1]
        if dtype in ['tif', 'tif', 'svs']:
            self.openslide = True
            self.handle = openslide.OpenSlide(filename)
            self._shape = self.handle.level_dimensions[0]
        else:
            self.openslide = False
            img = cv2.imread(filename)[:, :, ::-1] # To RGB
            h, w, _ = img.shape
            # openslide (width, height)
            self.img = img
            self._shape = [w, h]

    def read_region(self, location, level, size):
        # convert coors, the coors always on level 0
        x, y = location
        w, h = size
        _w = int(w*self.level_downsamples[level])
        _h = int(h*self.level_downsamples[level])

        if self.openslide:
            img = self.handle.read_region(location, 0, (_w, _h)).resize((w, h)).convert('RGB')
        else:
            img = self.img[y: y + _h, x: x + _w].copy()
            img = Image.fromarray(img).resize((w, h))
        return img
    
    def __read(self, location, level, size):
        w, h = size
        _w = int(w*self.level_downsamples[level])
        _h = int(h*self.level_downsamples[level])
        r = 1/self.level_downsamples[level]

        if _w < 20000 or _h < 20000:
            img = self.handle.read_region(location, 0, (_w, _h)).resize((w, h))
        else:
            step = 10000
            img = []
            x, y = location
            ex, ey = _w + x, _h + y
            xx = list(range(x, ex, step))
            xx = xx if ex in xx else xx + [ex]
            yy = list(range(y, ey, step))
            yy = yy if ey in yy else yy + [ey]
            # top to down
            counter = 0
            for _yy in yy:
                temp = []
                for _xx in xx:
                    t = np.array(self.handle.read_region((_xx, _yy), 0, (step, step)))
                    t = cv2.resize(t, None, fx=r, fy=r)
                    temp.append(t)
                    counter += 1
                    print(counter, len(yy)*len(xx))
                temp = np.concatenate(temp, axis=1)
                img.append(temp)

            img = np.concatenate(img, axis=0)
            img = Image.fromarray(img)
        return img


    @property
    def dimensions(self):
        return self.level_dimensions[0]

    @property
    def level_count(self):
        return len(self.default_dims)
    
    @property
    def level_downsamples(self):
        shape = [self._shape[0]/r[0] for r in self.level_dimensions]
        return shape
    
    @property
    def level_dimensions(self):
        shape = [(int(self._shape[0]/r), int(self._shape[1]/r)) for r in self.default_dims]
        return shape
    
    def get_best_level_for_downsample(self, scale):
        preset = [i*i for i in self.level_downsamples]
        err = [abs(i-scale) for i in preset]
        level = err.index(min(err))
        return level

    def close(self):
        pass