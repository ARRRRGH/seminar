from joblib import Parallel, delayed


def run_jobs(jobs, joblib=True, n_jobs=4):
    if joblib:
        jobs = [delayed(job)() for job in jobs]
        out = Parallel(n_jobs=n_jobs, backend='threading')(jobs)
    else:
        out = []
        for job in jobs:
            out.append(job())
    return out


class BiDict(dict):
    def __init__(self, *args, **kwargs):
        super(BiDict, self).__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse.setdefault(value,[]).append(key)

    def __setitem__(self, key, value):
        if key in self:
            self.inverse[self[key]].remove(key)
        super(BiDict, self).__setitem__(key, value)
        self.inverse.setdefault(value,[]).append(key)

    def __delitem__(self, key):
        self.inverse.setdefault(self[key],[]).remove(key)
        if self[key] in self.inverse and not self.inverse[self[key]]:
            del self.inverse[self[key]]
        super(BiDict, self).__delitem__(key)