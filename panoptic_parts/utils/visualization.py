import itertools
import random
import collections
import operator

from scipy import ndimage
import numpy as np

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

# Colormap for painting parts, similar to Matlab's parula(6) colormap.
# For the "void/unlabeled" semantic-instance-parts-level color, we use
#   the semantic-instance-level color is used instead of PARULA6[0]
N_MAX_COLORABLE_PARTS = 5
PARULA6 = [
    (61, 38, 168),
    (27, 170, 222), (71, 203, 134), (234, 186, 48), (249, 250, 20), (67, 102, 253)]


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
  _, _, pids = decode_uids(np.unique(np.array(uids, dtype=np.int32)))
  pid_max = np.amax(pids)
  if pid_max > N_MAX_COLORABLE_PARTS:
    raise NotImplementedError(
        f"Up to 5 parts are supported for coloring. Found pid={pid_max}.")


def uid2color(uids,
              sid2color=None,
              experimental_deltas=(60, 60, 60),
              experimental_alpha=0.5):
  """
  Create an RGB palette for all unique uids in `uids`. The palette is a dictionary mapping
  each uid from `uids` to an RGB color tuple, with values in range [0, 255].
  The uids have to comply with the hierarchical format (see README), i.e., uid = (sid, iid, pid).

  The colors are generated in the following way:
    - if uid represents a semantic-level label, i.e. uid=(sid, N/A, N/A),
      then `sid2color`[sid] is used.
    - if uid represents a semantic-instance-level label, i.e. uid=(sid, iid, N/A),
      then a random shade of `sid2color`[sid] is used, controlled by `experimental_deltas`.
      The shades are generated so they are as diverse as possible and the variability depends
      on the number of iids per sid, i.e., the more the instances per sid in the `uids`,
      the less the discriminability of shades.
    - if uid represents a semantic-instance-parts-level label, i.e. uid=(sid, iid, pid),
      then a random shade is generated as in the semantic-instance-level above and then
      it is mixed with a single color from the parula colormap, controlled by `experimental_alpha`.

  If `sid2color` is not provided (is None) then random colors are used. If `sid2color`
  is provided but does not contain all the sids of `uids` an error is raised.

  For now up to 5 parts per sid are supported, i.e., 1 <= pid <= 5.

  Example usage in cityscapes_panoptic_parts/experimental_visualize.py.

  Args:
    uids: a list of Python int, or an np.ndarray, with elements following the hierarchical labeling
      format defined in README
    sid2color: a dict mapping each sid of uids to an RGB color tuple of Python ints
      with values in range [0, 255], sids that are not present in uids will be ignored
    experimental_deltas: the range per color (Red, Green, Blue) in which to create shades, a small
      range provides shades that are close to the sid color but makes instance colors to have less
      contrast, a higher range provides better contrast but may create similar colors between
      different sid instances
    experimental_alpha: the mixing coeffient of the shade and the parula color, a higher value
      will make the semantic-instance-level shade more dominant over the parula color

  Returns:
    uid2color: a dict mapping each uid to a color tuple of Python ints in range [0, 255]
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

  ## generate the uid to colors mappings
  # convert set to list so it is indexable
  # the index is needed since iids do not need be to be continuous,
  #   otherwise we could just do sid2shades[sid][iid]
  sid_2_iids = {sid: list(iids) for sid, iids in _sid2iids(set(uids)).items()}
  uid_2_color = dict()
  for uid in set(uids):
    sid, iid, pid = decode_uids(uid)
    if uid <= 99:
      uid_2_color[uid] = sid2color[sid]
      continue
    index = sid_2_iids[sid].index(iid)
    sem_inst_level_color = sid2shades[sid][index]
    if uid <= 99_999 or pid == 0:
      uid_2_color[uid] = sem_inst_level_color
      continue
    if pid >= 1:
      uid_2_color[uid] = tuple(map(int,
          experimental_alpha * np.array(sem_inst_level_color) +
              (1-experimental_alpha) * np.array(PARULA6[pid])))

  return uid_2_color


def experimental_colorize_label(label,
                                sid2color=None,
                                return_sem=False,
                                return_sem_inst=False,
                                emphasize_instance_boundaries=True):
  """
  Colorizes a `label` with part-level panoptic colors (semantic-instance-parts-level)
  based on sid2color. See uid2color for more info.
  """
  
  assert all([isinstance(label, np.ndarray), label.ndim == 2, label.dtype == np.int32])

  # We visualize labels on three levels: semantic, semantic-instance, semantic-instance-parts.
  # We want to colorize same instances with the same shades across subfigures for easier comparison
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
  edge_option = 'sobel' # or 'erosion'
  if emphasize_instance_boundaries:
    # TODO(panos): simplify this algorithm
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
