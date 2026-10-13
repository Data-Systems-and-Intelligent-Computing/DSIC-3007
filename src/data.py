"""Research preparation and sampling entry points.

Source partitions require the reviewed source-group allocation in preparation.
The fresh study does not use the former random 50/50 document split.
"""

from .preparation import (eligible_positions, make_plan, prepare,
                          reviewed_documents, sample_positions, window_for)

select_evaluation_positions = sample_positions
