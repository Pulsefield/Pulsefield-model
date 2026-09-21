"""Typed candidate scheduling; absence never becomes an all-empty V3 row."""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Schedule:
    onsets: tuple
    index: int = 0
    ends: tuple = (None, None, None, None)

    @property
    def finished(self):
        return self.index == len(self.onsets)

    @property
    def next_onset(self):
        return next((j for j in range(self.index+1,len(self.onsets)) if self.onsets[j]),None)

    def row_possible(self, actions):
        if self.finished or len(actions)!=4:
            return False
        if bool(any(a in (1,2) for a in actions)) != bool(self.onsets[self.index]):
            return False
        future=[]
        for end,action in zip(self.ends,actions):
            if end is not None:
                if end<self.index or action!=(3 if end==self.index else 0):
                    return False
                future.append(None if end==self.index else end)
            else:
                if action not in (0,1,2) or (action==2 and self.index+1==len(self.onsets)):
                    return False
                future.append(self.index+1 if action==2 else None)
        following=self.next_onset
        return following is None or any(end is None or end<following for end in future)

    def commit(self, actions, selected_ends=None):
        actions=tuple(actions)
        selected_ends={} if selected_ends is None else selected_ends
        if not self.row_possible(actions):
            raise ValueError('Illegal typed head/release row or no feasible future onset')
        starts={c for c,a in enumerate(actions) if a==2}
        if set(selected_ends)!=starts:
            raise ValueError('Each selected LN head requires exactly one endpoint')
        ends=list(self.ends)
        for c,a in enumerate(actions):
            if a==3:
                ends[c]=None
            elif a==2:
                end=selected_ends[c]
                if type(end) is not int or not self.index<end<len(self.onsets):
                    raise ValueError('Endpoint must be a strictly future candidate')
                ends[c]=end
        following=self.next_onset
        if following is not None and not any(e is None or e<following for e in ends):
            raise ValueError('Endpoint assignment blocks the next required onset')
        result=replace(self,index=self.index+1,ends=tuple(ends))
        if result.finished and any(e is not None for e in ends):
            raise ValueError('Finished schedule retains a hold')
        return result, actions if any(actions) else None

    def forced_actions(self):
        if self.finished or self.onsets[self.index]:
            raise ValueError('Only optional non-onset slots are deterministic')
        return tuple(3 if e==self.index else 0 for e in self.ends)
