from panoptic_parts.utils.visualization import random_colors

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


if __name__ == "__main__":
  random_colors_test()