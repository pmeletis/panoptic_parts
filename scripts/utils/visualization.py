
import itertools
import random
import collections

from scipy import ndimage
import numpy as np

from scripts.utils.format import decode_uids

# Functions that start with underscore (_) should be considered as internal.
# All other functions belong to the public API.
# Arguments and functions defined with the preffix experimental_ may be changed
# and are not backward-compatible.

# PUBLIC_API = [colorize_uids]

def _generate_shades(center_color, deltas, num_of_shades):
  # center_color: (R, G, B)
  # deltas: (R ± ΔR, G ± ΔG, B ± ΔB)
  # returns a list of rgb color tuples
  # TODO: move all checks to the first API-visible function
  if num_of_shades <= 0:
    return []
  if num_of_shades == 1:
    return [center_color]
  if not all(map(lambda d: 0 <= d <= 255, deltas)):
    raise ValueError('deltas were not valid.')

  center_color = np.array(center_color)
  deltas = np.array(deltas)
  starts = np.maximum(0, center_color - deltas)
  stops = np.minimum(center_color + deltas + 1, 255)
  # in order to generate num_of_shades colors we divide the range
  #   by the cardinality of the cartesian product |R × G × B| = |R| · |G| · |B|,
  #   i.e. cbrt(num_of_shades)
  steps = np.floor((stops - starts) / np.ceil(np.cbrt(num_of_shades)))
  shades = itertools.product(*map(np.arange, starts, stops, steps))
  # convert to int
  shades = list(map(lambda shade: tuple(map(int, shade)), shades))

  # sanity check
  assert len(shades) >= num_of_shades, (
      f"_generate_shades: Report case with provided arguments as an issue.")

  return random.sample(shades, num_of_shades)

def _num_instances_per_sid(uids):
  # Note: instances in Cityscapes are not always labeled with continuous iids,
  # e.g. one image can have instances with iids: 000, 001, 003, 007
  uids = np.unique(uids)
  _, _, _, sids_iids = decode_uids(uids, experimental_return_sids_iids=True)
  sid2Ninstances = collections.defaultdict(lambda : 0)
  for sid_iid in sids_iids:
    sid, _, _ = decode_uids(sid_iid)
    sid2Ninstances[sid] += 1
  return sid2Ninstances

def _uid2colors(uids, id2color=None, experimental_deltas=(60, 60, 60), experimental_alpha=0.5):
  """
  Creates a palette for all unique uids in `uids`.

  Note: Only up to 6 parts per semantic class can be colored for now.
  An error will be thrown otherwise.

  Args:
    uids: a list of uids, following the hierarchical labeling format defined in README.md
    id2color: a dict mapping ids used for encoding the semantic class of
      uids (first two digits) to colors,
      shades of these colors will be generated for different instances and parts

  Returns:
    uid2colors: a dict mapping each uid to three colors:
      a class color, an instance color, and a part color.
      Colors for some uids may be the same depending on the existence or not of
      instances or parts annotations.
      uid2colors[uid][0] is just copied from id2color[uid],
      uid2colors[uid][1] is a random shade of uid2colors[uid][0],
      uid2colors[uid][2] is a mixture of uid2colors[uid][1] and the parula colormap
      Use uid2colors[uid][0] for a semantic-level coloring,
      uid2colors[uid][1] for semantic-instance-level coloring,
      uid2colors[uid][2] for semantic-instance-part-level coloring.

  """
  # TODO: move all checks to the first API-visible function
  if not isinstance(id2color, dict) or id2color is None:
    raise NotImplementedError('id2color must be a dict for now.')
  if experimental_alpha < 0 or experimental_alpha > 1:
    raise ValueError('experimental_alpha must be in [0, 1].')
  if np.any(uids < 0) or np.any(uids > 99_999_99):
    raise ValueError(f'There are uids that are not in the correct range\n{np.unique(uids)}.')

  N_MAX_COLORABLE_PARTS = 6
  # mixed parula so visualization is better
  parula6 = [
    #   (0, 0, 0),
      (61, 38, 168), (27, 170, 222), (71, 203, 134),
      (234, 186, 48), (249, 250, 20), (67, 102, 253)]

  # generate instance shades
  sid2num_instances = _num_instances_per_sid(uids)
  sid2shades = {sid: _generate_shades(id2color[sid], experimental_deltas, Ninstances)
                for sid, Ninstances in sid2num_instances.items()}

  # generate the uid to colors mappings
  uid_2_colors = dict()
  def _add_mapping(uid, cc, ic, pc):
    uid_2_colors[uid] = list(map(np.uint8, [cc, ic, pc]))

  for uid in np.unique(uids):
    sid, iid, pid = decode_uids(uid)

    # only semantic labels
    if uid <= 99:
      ccolor = id2color[uid]
      _add_mapping(uid, ccolor, ccolor, ccolor)
      continue

    # from this point onward we have at least semantic labels
    ccolor = id2color[sid]
    icolor = sid2shades[sid][iid]

    # only semantic and instance labels
    if uid <= 99_999:
      _add_mapping(uid, ccolor, icolor, icolor)
      continue

    # from this point onward we have at least semantic and instance labels

    # semantic, instance, and part labels
    if uid <= 99_999_99:
      if pid > N_MAX_COLORABLE_PARTS - 1:
        raise NotImplementedError(
            f'Up to 6 parts are supported for coloring. Found uid={uid}, pid={pid}.')
      pcolor = (experimental_alpha * np.array(icolor) +
                (1-experimental_alpha) * np.array(parula6[pid]))
      _add_mapping(uid, ccolor, icolor, pcolor)
      continue

  return uid_2_colors

def colorize_uids(uids, sid2color=None, experimental_emphasize_instance_boundary=False):
  """
  This function colorizes a `uids` array that has values according to the
  hierarchical format in the README. For the semantic level it uses the colors
  provided in `sid2color` and creates internally colors for the 
  instance level different instances (random shades of `sid2color`) and
  parts level different parts (a blend of parula color map and `sid2color`).

  Example usage: ....py

  Limitations: only up to 7 parts are supported for now (pids: 0-6)

  Args:
    uids: np.ndarray, np.int32, 2-D, with values according to hierarchical format in the README
    sid2color: Python dict, mapping the semantic class ids (sids) in uids to an RGB color tuple,
      with values in [0, 255],
    experimental_emphasize_instance_boundaries: Python boolean,
      if True will add a white boundary around instances (for now its experimental)

  Return:
    The following three np.ndarray, np.uint8, 3-D (RGB), with values in [0, 255]:
      uids_semantic_colored: `uids` colored on semantic level,
        using the colors in `sid2color`
      uids_instance_colored: `uids` colored on semantic and instance level,
        using shades of colors in `sid2color` to denote different instances
      uids_parts_colored: `uids` colored on semantic, instance and part level,
        using shades of colors in `sid2color` to denote different instances
        and mixing them with parula color palette for pixels with parts
  """

  edge_option = 'sobel' # or 'erosion'

  if sid2color is None:
    raise NotImplementedError('Random colors for sid2color will be supported in the future.')

  uid_2_colors = _uid2colors(np.unique(uids), sid2color)

  # create the palettes so advanced indexing for coloring can be used
  # initialize all colors in palettes with white (255, 255, 255) (for easier debugging)
  uids_keys = list(uid_2_colors.keys())
  # palettes for sids, iids, and pids
  palette_sids = np.full((np.max(uids_keys)+1, 3), 255, dtype=np.uint8)
  palette_iids = np.copy(palette_sids)
  palette_pids = np.copy(palette_sids)
  for uid, colors in uid_2_colors.items():
    palette_sids[uid] = colors[0]
    palette_iids[uid] = colors[1]
    palette_pids[uid] = colors[2]

  # create colored images using the palettes
  uids_sids_colored = palette_sids[uids]
  uids_iids_colored = palette_iids[uids]
  uids_pids_colored = palette_pids[uids]

  if experimental_emphasize_instance_boundary:
    _, iids, _ = decode_uids(uids)
    # create per-instance binary masks
    iids_unique = np.unique(iids)
    boundaries = np.full(iids.shape, False)
    edges = np.full(iids.shape, False)
    for iid in iids_unique:
      if 0 <= iid <= 999:
        iid_mask = np.equal(iids, iid)
        if edge_option == 'sobel':
          edge_horizont = ndimage.sobel(iid_mask, 0)
          edge_vertical = ndimage.sobel(iid_mask, 1)
          edges = np.logical_or(np.hypot(edge_horizont, edge_vertical), edges)
        elif edge_option == 'erosion':
          boundary = np.logical_xor(iid_mask,
                                    ndimage.binary_erosion(iid_mask, structure=np.ones((4, 4))))
          boundaries = np.logical_or(boundaries, boundary)

    if edge_option == 'sobel':
      boundaries_image = np.uint8(edges)[..., np.newaxis] * np.uint8([[[255, 255, 255]]])
    elif edge_option == 'erosion':
      boundaries_image = np.uint8(boundaries)[..., np.newaxis] * np.uint8([[[255, 255, 255]]])

    uids_iids_colored = np.where(boundaries_image,
                                 boundaries_image,
                                 uids_iids_colored)
    uids_pids_colored = np.where(boundaries_image,
                                 boundaries_image,
                                 uids_pids_colored)

  return uids_sids_colored, uids_iids_colored, uids_pids_colored
