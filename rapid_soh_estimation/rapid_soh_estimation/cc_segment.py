
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from config import *
from common_methods import *







if __name__ == '__main__':
    cc_segment_length = 600             # CC segment length (in seconds)
    cc_segment_overlap = 0.5            # overlap of CC segments when creating subsamples (0.0 = no overlap)
    cc_segment_soc_bounds = (30,90)     # use only CC data between these state-of-charge bounds

