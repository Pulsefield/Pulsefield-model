import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).parent.parent/'linked-endpoint-head-v1/scripts'))
from core import linked_rows
from schedule import Schedule
from ensomi_model.research.oracle_time_continuation.storage import ROW_DTYPE


class ScheduleChecks(unittest.TestCase):
    def test_source_objects_reconstruct_crossing_holds(self):
        rows=np.array([(0,[2,2,0,0]),(40,[0,0,1,0]),(100,[3,0,2,0]),
                       (160,[1,3,0,0]),(5000,[0,0,3,0])],dtype=ROW_DTYPE)
        onsets,ends,_=linked_rows(rows)
        state=Schedule(tuple(bool(x) for x in onsets))
        for i,actions in enumerate(rows['actions']):
            state,actual=state.commit(tuple(int(a) for a in actions),{int(c):int(ends[i,c]) for c in np.flatnonzero(actions==2)})
            self.assertEqual(actual,tuple(actions))
        self.assertTrue(state.finished);self.assertTrue(all(e is None for e in state.ends))

    def test_unused_candidates_have_no_row_including_terminal(self):
        state=Schedule((True,False,True,False));materialized=[]
        for action in [(1,0,0,0),(0,0,0,0),(0,1,0,0),(0,0,0,0)]:
            state,row=state.commit(action)
            if row is not None:materialized.append(row)
        self.assertEqual(len(materialized),2);self.assertTrue(state.finished)

    def test_future_onset_requires_strictly_earlier_free_lane(self):
        state=Schedule((True,False,True,False))
        with self.assertRaises(ValueError):state.commit((2,2,2,2),{i:3 for i in range(4)})
        state,_=state.commit((2,2,2,2),{0:1,1:3,2:3,3:3})
        state,row=state.commit(state.forced_actions());self.assertEqual(row,(3,0,0,0))
        state,_=state.commit((1,0,0,0))
        state,_=state.commit(state.forced_actions());self.assertTrue(state.finished)
        self.assertFalse(Schedule((True,True,False)).row_possible((2,2,2,2)))

    def test_release_and_restart_same_lane_is_illegal(self):
        state,_=Schedule((True,True,False)).commit((2,0,0,0),{0:1})
        self.assertFalse(state.row_possible((1,0,0,0)))
        state,row=state.commit((3,1,0,0));self.assertEqual(row,(3,1,0,0))


if __name__=='__main__':unittest.main()
