
import yaml

from panoptic_parts.utils.utils import parse_sids_pids_ppp_to_dense_mapping


m = yaml.load(open('panoptic_parts/utils/defs/ppp_60_57.yaml'))['sids_pids_ppp2sids_pids']
mnew = parse_sids_pids_ppp_to_dense_mapping(m)
# print(*sorted(mnew.items()), sep='\n')