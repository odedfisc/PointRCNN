import os
from pypeln import process as pr
from preprocessing.build.make.modules.features_extractor.features_extractor_py import FeaturesExtractor
from functools import partial
import numpy as np


def process(file, ext=None, data_path=None):
    if file.endswith(".bin"):
        frame = np.fromfile(os.path.join(data_path, file), dtype=np.float32).reshape(-1, 4)
        distances = np.ascontiguousarray(np.linalg.norm(frame[:, 0:3], axis=1))
        points = np.ascontiguousarray(frame[:, 0:3])
        dirs = np.ascontiguousarray(points / distances[:, None])
        ext.set_directions(dirs, False, True)
        features = ext.calc_simple_features(points, distances)
        features = features.reshape([-1, 6])
        print(np.var(features, axis=0))
        with open(os.path.join(data_path, file).split('.')[0] + "_features.npy", 'wb') as f:
            np.save(f, features)
            print(f"Write {file}")


if __name__ == '__main__':
    data_path = "/mnt/weka01/cvalgo/KITTI/object/training/velodyne_small"
    ext = FeaturesExtractor(0.4, 0.7)
    # for file in os.listdir(data_path):
    #     process(file, ext=ext, data_path=data_path)
    stage = pr.map(partial(process, ext=ext, data_path=data_path), os.listdir(data_path), workers=14)
    process_iterator = pr.to_iterable(stage)
    while True:
        try:
            features_df_tmp = process_iterator.__next__()
        except StopIteration:
            break



