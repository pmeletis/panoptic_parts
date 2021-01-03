import itertools
import random
import collections
import operator
import functools

from scipy import ndimage
import numpy as np
import matplotlib

from panoptic_parts.utils.format import decode_uids
from panoptic_parts.utils.utils import _sparse_ids_mapping_to_dense_ids_mapping

# Functions that start with underscore (_) should be considered as internal.
# All other functions belong to the public API.
# Arguments and functions defined with the preffix experimental_ may be changed
# and are not backward-compatible.

# PUBLIC_API = [random_colors, uid2color]

# TODO(panos): Make VALIDATE_ARGS global. Exhaustive validation is usually computationally
#   intensive, so we want to have a global switch (which late will default to False),
#   that during debugging only turns argument validation on
VALIDATE_ARGS = True

# For Cityscapes Panoptic Parts the previously defined parula colormap slightly differs from
# PARULA99. This is done so vehicle chassis is colored with blue shades and thus resemble the
# original colors. This flag is enabled by default, although, if it is not possible to use the
# legacy colormap PARULA99 colormap is used. Otherwise, use set_use_legacy_cpp_parts_colormap.
# This flag will be disabled by default in the future.
USE_LEGACY_CPP_PARTS_COLORMAP = True
def set_use_legacy_cpp_parts_colormap(boolean):
  global USE_LEGACY_CPP_PARTS_COLORMAP
  assert isinstance(boolean, bool)
  USE_LEGACY_CPP_PARTS_COLORMAP = boolean
# same as parula99_cm(np.linspace(0, 1, 6)), but with second color (id=1) moved to the end
LEGACY_PARULA6 = [
    # (61, 38, 168),
    (27, 170, 222), (71, 203, 134), (234, 186, 48), (249, 250, 20), (67, 102, 253)]

# MATLAB® PARULA99 colormap, generated with Matlab 2019a: uint8(floor(parula(99)*255))
# This colormap is used for colorizing up to 99 parts pids
PARULA99_INT = [
    (61, 38, 168), (63, 40, 176), (64, 43, 183), (65, 46, 190), (66, 48, 197),
    (67, 51, 205), (68, 54, 211), (69, 57, 217), (70, 60, 223), (70, 64, 227),
    (71, 67, 231), (71, 71, 235), (71, 75, 238), (71, 78, 241), (71, 82, 244),
    (71, 85, 246), (71, 89, 248), (70, 93, 250), (69, 96, 251), (68, 100, 252),
    (66, 104, 253), (64, 108, 254), (61, 112, 254), (57, 116, 254), (53, 120, 253),
    (49, 124, 252), (47, 127, 250), (46, 131, 248), (45, 134, 246), (45, 138, 244),
    (44, 141, 241), (43, 145, 238), (40, 148, 236), (38, 151, 234), (37, 154, 231),
    (36, 157, 230), (34, 160, 228), (32, 163, 227), (30, 166, 225), (28, 169, 223),
    (25, 172, 220), (22, 174, 217), (17, 177, 214), (11, 179, 210), (4, 181, 206),
    (1, 183, 202), (0, 185, 198), (2, 187, 193), (9, 188, 189), (17, 190, 185),
    (26, 191, 180), (33, 192, 175), (39, 194, 171), (44, 195, 166), (47, 197, 161),
    (51, 198, 156), (55, 199, 151), (59, 201, 145), (65, 202, 139), (73, 203, 133),
    (81, 203, 126), (89, 204, 119), (97, 204, 112), (106, 204, 105), (115, 204, 98),
    (125, 204, 91), (134, 203, 84), (144, 202, 76), (153, 201, 69), (162, 200, 62),
    (171, 198, 56), (179, 197, 51), (188, 195, 46), (196, 193, 42), (204, 192, 39),
    (211, 190, 39), (218, 189, 40), (225, 187, 42), (231, 186, 46), (237, 185, 51),
    (243, 185, 57), (248, 186, 61), (252, 188, 61), (254, 191, 58), (254, 195, 56),
    (254, 199, 53), (253, 202, 50), (252, 207, 48), (250, 211, 46), (248, 215, 44),
    (247, 219, 42), (245, 223, 40), (244, 227, 38), (244, 231, 36), (244, 235, 34),
    (245, 239, 31), (246, 243, 28), (247, 247, 24), (249, 250, 20)]
PARULA99_FLOAT = list(map(lambda t: tuple(map(lambda c: c/255, t)), PARULA99_INT))
# parula_cm(x), x can be float in [0.0, 1.0] or int in [0, 99) to return a color
PARULA99_CM = matplotlib.colors.LinearSegmentedColormap.from_list('parula99', PARULA99_FLOAT, 99)


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
    raise ValueError('Provide a correct, Python int number of colors.')

  return [tuple(map(int, color)) for color in np.random.choice(256, size=(num, 3))]


def _generate_shades(center_color, deltas, num_of_shades):
  # center_color: (R, G, B)
  # deltas: (R ± ΔR, G ± ΔG, B ± ΔB)
  # returns a list of rgb color tuples
  # TODO: move all checks to the first API-visible function
  if num_of_shades <= 0:
    raise ValueError(f"num_of_shades must be a positive integer (was {num_of_shades}).")
  if num_of_shades == 1:
    return [center_color]
  # TODO: enable d=0
  if not all(map(lambda d: 0 < d <= 255, deltas)):
    raise ValueError(f"deltas were not valid ({deltas}).")

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

  return list(sorted(random.sample(shades, num_of_shades), key=lambda t: np.linalg.norm(t, ord=2)))


def _num_instances_per_sid(uids):
  # Note: instances in Cityscapes are not always labeled with continuous iids,
  # e.g. one image can have instances with iids: 000, 001, 003, 007
  # TODO(panos): move this functionality to utils.format
  # np.array is needed since uids are Python ints
  # and np.unique implicitly converts them to np.int64
  # TODO(panos): remove this need when np.int64 is supported in decode_uids
  uids_unique = np.unique(np.array(uids, dtype=np.int32))
  _, _, _, sids_iids = decode_uids(uids_unique, return_sids_iids=True)
  sids_iids_unique = np.unique(sids_iids)
  sid2Ninstances = collections.defaultdict(lambda : 0)
  for sid_iid in sids_iids_unique:
    sid, iid, _ = decode_uids(sid_iid)
    if iid >= 0:
      sid2Ninstances[sid] += 1
  return sid2Ninstances


def _num_parts_per_sid(uids):
  assert isinstance(uids, list)
  # TODO(panos): add the list decoding functionality in decode_uids
  sids_pids_unique = set(
      map(operator.itemgetter(3),
          map(functools.partial(decode_uids, return_sids_pids=True), uids)))
  sid2Nparts = collections.defaultdict(lambda : 0)
  for sid_pid in sids_pids_unique:
    sid_pid_full = sid_pid * 100 if sid_pid <= 99 else sid_pid
    sid = sid_pid_full // 100
    pid = sid_pid_full % 100
    if pid > 0:
      sid2Nparts[sid] += 1
  return sid2Nparts


def _sid2iids(uids):
  # a dict mapping a sid to a set of all its iids
  # or in other words a mapping from a semantic class to all object ids it has
  # uids: a list of Python int uids
  # iids do not need to be consecutive numbers
  # TODO(panos): move this functionality to utils.format
  sid2iids = collections.defaultdict(set)
  for uid in set(uids):
    sid, iid, _ = decode_uids(uid)
    # decode_uids returns iid = -1 for pixels that don't have instance-level labels
    if iid >= 0:
      sid2iids[sid].add(iid)
  return sid2iids


def _sid2pids(uids):
  # a dict mapping a sid to a set of all its pids
  # uids: a list of Python int uids
  # TODO(panos): move this functionality to utils.format
  assert isinstance(uids, list)
  sid2pids = collections.defaultdict(set)
  for uid in set(uids):
    sid, _, pid = decode_uids(uid)
    # decode_uids returns pid = -1 for pixels that don't have part-level labels
    if pid >= 0:
      sid2pids[sid].add(pid)
  return sid2pids
  

def _validate_uid2color_args(uids, sid2color, experimental_deltas, experimental_alpha):
  # TODO(panos): add more checks for type, dtype, range
  # TODO(panos): optimize performance by minimizing overlapping functionality
  if not isinstance(uids, (list, np.ndarray)):
    raise ValueError(f"Provide a list or np.ndarray of uids. Given {type(uids)}.")
  if isinstance(uids, np.ndarray):
    uids = list(map(int, np.unique(uids)))
  if not all(map(isinstance, uids, [int]*len(uids))):
    raise ValueError(f"Provide a list of Python ints as uids. Given {uids}.")
  if not all(map(lambda uid: 0 <= uid <= 99_999_99, uids)):
    raise ValueError(f'There are uids that are not in the correct range. Given {uids}.')
  # sid2color checks
  if not isinstance(sid2color, dict) and sid2color is not None:
    raise ValueError(f"sid2color must be a dict. Given {type(sid2color)}.")
  sids_unique_from_uids = set(map(operator.itemgetter(0), map(decode_uids, uids)))
  if not sids_unique_from_uids.issubset(sid2color):
    raise ValueError(f"Not all sids in uids have a matching color in sid2color.")
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
  # _, _, pids = decode_uids(np.unique(np.array(uids, dtype=np.int32)))
  # pid_max = np.amax(pids)
  # if pid_max > N_MAX_COLORABLE_PARTS:
  #   raise NotImplementedError(
  #       f"Up to 5 parts are supported for coloring. Found pid={pid_max}.")


def uid2color(uids,
              sid2color=None,
              experimental_deltas=(60, 60, 60),
              experimental_alpha=0.5):
  """
  Generate an RGB palette for all unique uids in `uids`. The palette is a dictionary mapping
  each uid from `uids` to an RGB color tuple, with values in range [0, 255].
  A uid is an up to 7-digit integer that is interpreted according to our panoptic parts format
  (see README), i.e., decode_uids(uid) = (sid, iid, pid).

  The colors are generated in the following way:
    - if uid represents a semantic-level label, i.e. uid = (sid, N/A, N/A),
      then `sid2color`[sid] is used.
    - if uid represents a semantic-instance-level label, i.e. uid = (sid, iid, N/A),
      then a random shade of `sid2color`[sid] is generated, controlled by `experimental_deltas`.
      The shades are generated so they are as diverse as possible and the variability depends
      on the number of iids per sid. The more the instances per sid in the `uids`, the less
      discriminable the shades are.
    - if uid represents a semantic-instance-parts-level label, i.e. uid = (sid, iid, pid),
      then a random shade is generated as in the semantic-instance-level case above and then
      it is mixed with a single color from the parula colormap, controlled by `experimental_alpha`.
      A different parula colormap is generated for each sid to achieve best discriminability
      of parts colors per sid.

  If `sid2color` is not provided (is None) then random colors are used. If `sid2color`
  is provided but does not contain all the sids of `uids` an error is raised.

  Example usage in {cityscapes, pascal}_panoptic_parts/visualize_from_paths.py.

  Args:
    uids: a list of Python int, or a np.int32 np.ndarray, with elements following the panoptic
      parts format (see README)
    sid2color: a dict mapping each sid of uids to an RGB color tuple of Python ints
      with values in range [0, 255], sids that are not present in uids will be ignored
    experimental_deltas: the range per color (Red, Green, Blue) in which to create shades, a small
      range provides shades that are close to the sid color but makes instance colors to have less
      contrast, a higher range provides better contrast but may create similar colors between
      different sid instances
    experimental_alpha: the mixing coeffient of the shade and the parula color, a higher value
      will make the semantic-instance-level shade more dominant over the parula color

  Returns:
    uid2color: a dict mapping each uid to a color tuple of Python int in range [0, 255]
  """

  if VALIDATE_ARGS:
    _validate_uid2color_args(uids, sid2color, experimental_deltas, experimental_alpha)

  if isinstance(uids, np.ndarray):
    uids = list(map(int, np.unique(uids)))

  ## generate semantic-level colors
  if sid2color is None:
    # TODO(panos): add the list decoding functionality in decode_uids
    sids_unique = set(map(operator.itemgetter(0), map(decode_uids, uids)))
    random_sids_palette = random_colors(len(sids_unique))
    sid2color = {sid: tuple(map(int, color))
                 for sid, color in zip(sids_unique, random_sids_palette)}

  ## generate instance shades
  sid2num_instances = _num_instances_per_sid(uids)
  # TODO(panos): experimental_deltas must be large for sids with many iids and small for
  #   sids with few iids, maybe automate this?
  sid2shades = {sid: _generate_shades(sid2color[sid], experimental_deltas, Ninstances)
                for sid, Ninstances in sid2num_instances.items()}

  ## generate discriminable per-sid parula colormap for parts
  # For best part-level color discriminability we generate a colormap per-sid,
  # this creates a discrininable colormap per-sid irrespectible of the number of parts.
  sid2num_parts = _num_parts_per_sid(uids)
  is_maybe_cpp = (USE_LEGACY_CPP_PARTS_COLORMAP and
                  all(map(lambda n: n<= 5, sid2num_parts.values())))
  sid2parulaX = {
      sid: LEGACY_PARULA6 if is_maybe_cpp
           else (PARULA99_CM(np.linspace(0, 1, num=Nparts))*255)[:,:3].astype(np.int32)
      for sid, Nparts in sid2num_parts.items()}

  ## generate the uid to colors mappings
  # convert sets to lists so they are indexable, the .index() is needed since iids and
  # pids do not need be to be continuous (otherwise sid2shades[sid][iid] is enough)
  # TODO(panos): sid_2_* have overlapping functionality, consider merging them
  sid_2_iids = {sid: list(iids) for sid, iids in _sid2iids(set(uids)).items()}
  def _remove_all_no_error(lst, el):
    if el in lst:
      lst.remove(el)
    assert el not in lst
    return lst
  sid_2_non_zero_pids = {sid: _remove_all_no_error(list(pids), 0)
                         for sid, pids in _sid2pids(uids).items()}

  uid_2_color = dict()
  for uid in set(uids):
    sid, iid, pid = decode_uids(uid)
    if uid <= 99:
      uid_2_color[uid] = sid2color[sid]
      continue
    index_iid = sid_2_iids[sid].index(iid)
    sem_inst_level_color = sid2shades[sid][index_iid]
    if uid <= 99_999 or pid == 0:
      uid_2_color[uid] = sem_inst_level_color
      continue
    if pid >= 1:
      index_pid = sid_2_non_zero_pids[sid].index(pid)
      uid_2_color[uid] = tuple(map(int,
          experimental_alpha * np.array(sem_inst_level_color) +
              (1-experimental_alpha) * np.array(sid2parulaX[sid][index_pid])))
    # catch any possible errors
    assert uid in uid_2_color.keys()

  return uid_2_color


def experimental_colorize_label(label,
                                sid2color=None,
                                return_sem=False,
                                return_sem_inst=False,
                                emphasize_instance_boundaries=True):
  """
  Colorizes a `label` with semantic-instance-parts-level colors based on sid2color.
  Optionally, semantic-level and semantic-instance-level colorings can be returned.
  The option emphasize_instance_boundaries will draw a 4-pixel white line around instance
  boundaries for the semantic-instance-level and semantic-instance-parts-level outputs.
  If a sid2color dict is provided colors from that will be used otherwise random colors
  will be generated.
  See panoptic_parts.utils.visualization.uid2color for how colors are generated.

  Args:
    label: 2-D, np.int32, np.ndarray with up to 7-digit uids, according to format in README
    sid2color: a dictionary mapping sids to RGB color tuples in [0, 255], all sids in `labels`
      must be in `sid2color`, otherwise provide None to use random colors
    return_sem: if True returns `sem_colored`
    return_sem_inst: if True returns `sem_inst_colored`

  Returns:
    sem_inst_parts_colored: 3-D, np.ndarray with RGB colors in [0, 255],
      colorized `label` with colors that distinguish scene-level semantics, part-level semantics,
      and instance-level ids
    sem_colored: 3-D, np.ndarray with RGB colors in [0, 255], returned if return_sem=True,
      colorized `label` with colors that distinguish scene-level semantics
    sem_inst_colored: 3-D, np.ndarray with RGB colors in [0, 255], returned if return_sem_inst=True,
      colorized `label` with colors that distinguish scene-level semantics and part-level semantics
  """
  if not isinstance(label, np.ndarray):
    raise ValueError(f"label is type: {type(label)}, only np.ndarray is supported.")
  if not all([label.ndim == 2, label.dtype == np.int32]):
    raise ValueError(
        f"label has: {label.ndim} dims and {label.dtype} dtype, only 2 dims"
        " and np.int32 are supported.")

  # We visualize labels on three levels: semantic, semantic-instance, semantic-instance-parts.
  # We want to colorize same instances with the same shades across levels for easier comparison
  # so we create ids_all_levels_unique and call uid2color() once to achieve that.
  # sids, iids, sids_iids shapes: (height, width)
  sids, iids, _, sids_iids = decode_uids(label, return_sids_iids=True)
  ids_all_levels_unique = np.unique(np.stack([sids, sids_iids, label]))
  uid2color_dict = uid2color(ids_all_levels_unique, sid2color=sid2color)

  # We colorize ids using numpy advanced indexing (gathering). This needs an array palette, thus we
  # convert the dictionary uid2color_dict to an array palette with shape (Ncolors, 3) and
  # values in range [0, 255] (RGB).
  # uids_*_colored shapes: (height, width, 3)
  palette = _sparse_ids_mapping_to_dense_ids_mapping(uid2color_dict, (0, 0, 0), dtype=np.uint8)
  uids_sem_colored = palette[sids]
  uids_sem_inst_colored = palette[sids_iids]
  uids_sem_inst_parts_colored = palette[label]

  # optionally add boundaries to the colorized labels uids_*_colored
  # TODO(panos): instance boundaries are found by the iids, if two iids are the same
  #   then an instance boundary is not drawn between different semantic-level classes
  # TODO(panos): same iids islands, that are occluded, must not have closed boundaries
  #   investigate if a solution to that is easy
  edge_option = 'sobel' # or 'erosion'
  if emphasize_instance_boundaries:
    # TODO(panos): simplify this algorithm
    # create per-instance binary masks
    sids_iids_unique = np.unique(sids_iids)
    boundaries = np.full(sids_iids.shape, False)
    edges = np.full(sids_iids.shape, False)
    for sid_iid in sids_iids_unique:
      iid = sid_iid % 1000
      if 0 <= iid <= 999:
        sid_iid_mask = np.equal(sids_iids, sid_iid)
        if edge_option == 'sobel':
          edge_horizont = ndimage.sobel(sid_iid_mask, 0)
          edge_vertical = ndimage.sobel(sid_iid_mask, 1)
          edges = np.logical_or(np.hypot(edge_horizont, edge_vertical), edges)
        elif edge_option == 'erosion':
          boundary = np.logical_xor(sid_iid_mask,
                                    ndimage.binary_erosion(sid_iid_mask, structure=np.ones((4, 4))))
          boundaries = np.logical_or(boundaries, boundary)

    if edge_option == 'sobel':
      boundaries_image = np.uint8(edges)[..., np.newaxis] * np.uint8([[[255, 255, 255]]])
    elif edge_option == 'erosion':
      boundaries_image = np.uint8(boundaries)[..., np.newaxis] * np.uint8([[[255, 255, 255]]])

    uids_sem_inst_colored = np.where(boundaries_image,
                                     boundaries_image,
                                     uids_sem_inst_colored)
    uids_sem_inst_parts_colored = np.where(boundaries_image,
                                           boundaries_image,
                                           uids_sem_inst_parts_colored)
    
  returns = (uids_sem_inst_parts_colored,)
  if return_sem:
    returns += (uids_sem_colored,)
  if return_sem_inst:
    returns += (uids_sem_inst_colored,)
  if len(returns) == 1:
    return returns[0]
  return returns
