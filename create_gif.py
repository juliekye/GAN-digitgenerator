import os
import imageio

images = []
filenames = os.listdir('snapshots')
filenames.sort(key=lambda x: int(x.replace('.png', '')))
for filename in filenames:
    images.append(imageio.imread('snapshots/' + filename))
imageio.mimsave('progress.gif', images)