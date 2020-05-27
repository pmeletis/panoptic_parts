
import itertools
import random
import collections
import operator

from scipy import ndimage
import numpy as np

from utils.format import decode_uids

# Functions that start with underscore (_) should be considered as internal.
# All other functions belong to the public API.
# Arguments and functions defined with the preffix experimental_ may be changed
# and are not backward-compatible.

# PUBLIC_API = [random_colors, uid2color]


def random_colors(num):
  """
  Returns a list of `num` random Python int RGB color tuples in range [0, 255].
  Colors can be repeated. This is desired behavior so we don't run out of colors.

  Args:
    num: Python int, the number of colors to produce

  Returns:
    colors: a list of tuples representing RGB colors in range [0, 255]
  """
  if not isinstance(num, int) or num < 0:
    raise ValueError('Provide a correct number of colors.')

  return [tuple(map(int, color)) for color in np.random.choice(256, size=(num, 3))]


def uid2color(uids,
              sid2color=None,
              experimental_deltas=(60, 60, 60),
              experimental_alpha=0.5):
  """
  Create an RGB palette for all unique uids in `uids`. The palette is a dictionary mapping
  each uid from `uids` to an RGB color tuple, with values in range [0, 255].
  The uids have to comply with the hierarchical format (see README), i.e., uid = (sid, iid, pid).

  The colors are generated in the following way:
    - If uid represents a semantic-level label (sid, N/A, N/A), then `sid2color`[sid] is used.
    - If uid represents a semantic-instance-level label (sid, iid, N/A), then a random shade
      of `sid2color`[sid] is used, controlled by `experimental_deltas`. The shades are
      generated so they are as diverse as possible and the variability depends on the number
      of iids per sid, i.e., the more the instances per sid in the `uids`, the less the
      discriminability of shades.
    - If uid represents a semantic-instance-parts-level label (sid, iid, pid), then a random shade
      is generated as in the semantic-instance-level above and then it is mixed with a single
      color from the parula colormap, controlled by `experimental_alpha`.

  If `sid2color` is not provided (is None) then random colors are used. If `sid2color`
  is provided but does not contain all the sids of uids an error is raised.

  For now up to 5 parts per sid are supported, i.e., 1 <= pid <= 5.

  Example usage in cityscapes_panoptic_parts/experimental_visualize.py.

  Args:
    uids: a list of Python int uids, following the hierarchical labeling format defined in README
    sid2color: a dict mapping each sid of uids to an RGB color tuple of Python ints
      with values in range [0, 255]
    experimental_deltas: the range per color (Red, Green, Blue) in which to create shades, a small
      range provides shades that are close to the sid color but makes instance colors to have less
      contrast, a higher range provides better contrast but may create similar colors between
      different sid instances
    experimental_alpha: the mixing coeffient of the shade and the parula color, a higher value
      will make the semantic-instance-level shade more dominant over the parula color

  Returns:
    uid2color: a dict mapping each uid to a color tuple of Python ints in range [0, 255]
  """
  # TODO(panos): add more checks for type, dtype, range

  # The colormap is similar to Matlab's parula(6) colormap.
  # By convention for the "void/unlabeled" semantic-instance-parts-level color,
  #   the semantic-instance-level color is used instead of PARULA6[0]
  N_MAX_COLORABLE_PARTS = 5
  PARULA6 = [
      (61, 38, 168),
      (27, 170, 222), (71, 203, 134), (234, 186, 48), (249, 250, 20), (67, 102, 253)]

  # Argument checking
  # uids checks
  if not isinstance(uids, list):
    raise ValueError(f"Provide a list for uids. Given {type(uids)}.")
  if not all(map(isinstance, uids, [int]*len(uids))):
    raise ValueError(f"Provide a list of Python ints as uids. Given {uids}.")
  if not all(map(lambda uid: 0 <= uid <= 99_999_99, uids)):
    raise ValueError(f'There are uids that are not in the correct range. Given {uids}.')
  # sid2color checks
  if not isinstance(sid2color, dict) and sid2color is not None:
    raise ValueError(f"sid2color must be a dict. Given {type(sid2color)}.")
  # experimental_deltas checks
  if not isinstance(experimental_deltas, tuple):
    raise ValueError(f"experimental_deltas must be a tuple. Given {type(experimental_deltas)}.")
  # if (not len(experimental_deltas) == 3 or
  #     not all(map(isinstance, experimental_deltas, [int]*len(experimental_deltas)))):
  #   raise
  # if not all(map(lambda c: 0 <= c <= 255, experimental_deltas)):
  #   raise
  # experimental_alpha checks
  if experimental_alpha < 0 or experimental_alpha > 1:
    raise ValueError('experimental_alpha must be in [0, 1].')
  # max pids check
  # we use np.array since uids are Python ints and np.unique implicitly converts them to np.int64
  # TODO(panos): remove this requirement when np.int64 is supported in decode_uids
  _, _, pids = decode_uids(np.unique(np.array(uids, dtype=np.int32)))
  pid_max = np.amax(pids)
  if pid_max > N_MAX_COLORABLE_PARTS:
    raise NotImplementedError(
        f"Up to 5 parts are supported for coloring. Found pid={pid_max}.")

  if sid2color is None:
    # TODO(panos): add the list decoding functionality in decode_uids
    sids_unique = set(map(operator.itemgetter(0), map(decode_uids, uids)))
    random_sids_palette = random_colors(len(sids_unique))
    sid2color = {sid: tuple(map(int, color))
                 for sid, color in zip(sids_unique, random_sids_palette)}

  # generate instance shades
  sid2num_instances = _num_instances_per_sid_v2(uids)
  # TODO(panos): experimental_deltas must be large for sids with many iids and small for
  #   sids with few iids, maybe automate this?
  sid2shades = {sid: _generate_shades(sid2color[sid], experimental_deltas, Ninstances)
                for sid, Ninstances in sid2num_instances.items()}

  # generate the uid to colors mappings
  # the index is needed since iids do not need be to be continuous,
  #   otherwise we could just do sid2shades[sid][iid]
  # convert set to list so it is indexable
  sid_2_iids = {sid: list(iids) for sid, iids in _sid2iids(set(uids)).items()}
  uid_2_colors = dict()
  for uid in set(uids):
    sid, iid, pid = decode_uids(uid)
    if uid <= 99:
      uid_2_colors[uid] = sid2color[sid]
    else:
      index = sid_2_iids[sid].index(iid)
      sem_inst_level_color = sid2shades[sid][index]
      if uid <= 99_999:
        uid_2_colors[uid] = sem_inst_level_color
      else:
        if pid >= 1:
          sem_inst_parts_level_color = tuple(map(int,
              experimental_alpha * np.array(sem_inst_level_color) +
                  (1-experimental_alpha) * np.array(PARULA6[pid])))
        else:
          sem_inst_parts_level_color = sem_inst_level_color
        uid_2_colors[uid] = sem_inst_parts_level_color

  return uid_2_colors


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
  # TODO(panos): deprecate this function since it has some bugs,
  #   use _num_instances_per_sid_v2 instead
  uids = np.unique(uids)
  _, _, _, sids_iids = decode_uids(uids, experimental_return_sids_iids=True)
  sid2Ninstances = collections.defaultdict(lambda : 0)
  for sid_iid in sids_iids:
    sid, _, _ = decode_uids(sid_iid)
    sid2Ninstances[sid] += 1
  return sid2Ninstances

def _num_instances_per_sid_v2(uids):
  # Note: instances in Cityscapes are not always labeled with continuous iids,
  # e.g. one image can have instances with iids: 000, 001, 003, 007
  # TODO(panos): move this functionality to utils.format
  # np.array is needed since uids are Python ints
  # and np.unique implicitly converts them to np.int64
  # TODO(panos): remove this need when np.int64 is supported in decode_uids
  uids_unique = np.unique(np.array(uids, dtype=np.int32))
  _, _, _, sids_iids = decode_uids(uids_unique, experimental_return_sids_iids=True)
  sids_iids_unique = np.unique(sids_iids)
  sid2Ninstances = collections.defaultdict(lambda : 0)
  for sid_iid in sids_iids_unique:
    sid, iid, _ = decode_uids(sid_iid)
    if iid >= 0:
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
      Use uid2colors[uid][0] for a semantic-level coloring,
      uid2colors[uid][1] for semantic-instance-level coloring,
      uid2colors[uid][2] for semantic-instance-part-level coloring.
      Colors for some uids may be the same depending on the existence or not of
      instances or parts annotations.
      uid2colors[uid][0] is just copied from id2color[uid],
      uid2colors[uid][1] is a random shade of uid2colors[uid][0],
      uid2colors[uid][2] is a mixture of uid2colors[uid][1] and the parula colormap.
  """
  # TODO(panos): the void part-level class (pid=0) must be colored with icolor
  #   and not parula6[0]
  # TODO: move all checks to the first API-visible function
  if not isinstance(id2color, dict) or id2color is None:
    raise NotImplementedError('id2color must be a dict for now.')
  if experimental_alpha < 0 or experimental_alpha > 1:
    raise ValueError('experimental_alpha must be in [0, 1].')
  if np.any(uids < 0) or np.any(uids > 99_999_99):
    raise ValueError(f'There are uids that are not in the correct range\n{np.unique(uids)}.')

  # TODO(panos): generate parula on the fly so more parts can be colored
  N_MAX_COLORABLE_PARTS = 6
  # parula colormap from Matlab
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
  def _add_mapping(uid, sc, ic, pc):
    uid_2_colors[uid] = list(map(np.uint8, [sc, ic, pc]))

  for uid in np.unique(uids):
    sid, iid, pid = decode_uids(uid)

    # only semantic labels
    if uid <= 99:
      scolor = id2color[uid]
      _add_mapping(uid, scolor, scolor, scolor)
      continue

    # from this point onward we have at least semantic labels
    scolor = id2color[sid]
    icolor = sid2shades[sid][iid]

    # only semantic and instance labels
    if uid <= 99_999:
      _add_mapping(uid, scolor, icolor, icolor)
      continue

    # from this point onward we have at least semantic and instance labels

    # semantic, instance, and part labels
    if uid <= 99_999_99:
      if pid > N_MAX_COLORABLE_PARTS - 1:
        raise NotImplementedError(
            f'Up to 6 parts are supported for coloring. Found uid={uid}, pid={pid}.')
      pcolor = (experimental_alpha * np.array(icolor) +
                (1-experimental_alpha) * np.array(parula6[pid]))
      _add_mapping(uid, scolor, icolor, pcolor)
      continue

  return uid_2_colors

def _colorize_uids(uids, sid2color=None, experimental_emphasize_instance_boundary=False):
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

def _sid2iids(uids):
  # a dict mapping a sid to a set of all its iids
  # uids: a list of Python int uids
  # iids do not need to be consecutive numbers
  # TODO(panos): move this functionality to utils.format
  sid2iids = collections.defaultdict(set)
  for uid in set(uids):
    sid, iid, _ = decode_uids(uid)
    if iid >= 0:
      sid2iids[sid].add(iid)
  return sid2iids
