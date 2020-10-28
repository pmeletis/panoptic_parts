import matplotlib as mpl
import matplotlib.pyplot as plt

from panoptic_parts.utils.visualization import random_colors, PARULA99_CM

def random_colors_test():
  colors0 = random_colors(0)
  assert len(colors0) == 0
  print(colors0)
  colors1 = random_colors(1)
  assert len(colors1) == 1
  print(colors1)
  colors10 = random_colors(10)
  assert len(colors10) == 10
  assert all(isinstance(color, tuple) for color in colors10)
  print(colors10)


def parula99_cm_test():
  # just a demo function plotting the colormap
  fig, ax = plt.subplots(figsize=(6, 1))
  fig.subplots_adjust(bottom=0.5)
  norm = mpl.colors.Normalize(vmin=1, vmax=PARULA99_CM.N + 1)
  fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=PARULA99_CM),
               cax=ax, orientation='horizontal', label='part-level semantic classes')
  fig.waitforbuttonpress(30.0)
  Nparts = 5
  bounds = list(range(1, Nparts + 1))
  norm = mpl.colors.BoundaryNorm(bounds, PARULA99_CM.N, extend='both')
  print(*map(norm, range(Nparts + 1 + 1)))
  mpl.colorbar.ColorbarBase(ax, cmap=PARULA99_CM, norm=norm, orientation='horizontal')
  plt.draw()
  fig.waitforbuttonpress(30.0)


if __name__ == "__main__":
  random_colors_test()
  parula99_cm_test()
