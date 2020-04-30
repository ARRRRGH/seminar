from joblib import Parallel, delayed

def run_jobs(jobs, joblib=True, n_jobs=4, chunks=1, chunk_callback=None, *args, **kwargs):
    if len(jobs) == 0:
        return None, None

    if joblib:
        jobs = [delayed(job)() for job in jobs]

        chunk_size = max(1, len(jobs) // chunks)
        chunks = [jobs[i:i + chunk_size] for i in range(0, len(jobs), chunk_size)]

        out = []
        for chunk in chunks:
            chunk_out = Parallel(n_jobs=n_jobs, *args, **kwargs)(chunk)
            if chunk_callback is not None:
                ret = chunk_callback(chunk_out, args=[job[0].args for job in chunk],
                                     kwargs=[job[0].keywords for job in chunk])
                out.append((chunk_out, ret))
            else:
                out.append((chunk_out, ))
    else:
        out = []
        # create chunks
        nr_chunks = chunks
        chunk_size = max(1, len(jobs) // chunks)
        chunks = [jobs[i:i + chunk_size] for i in range(0, len(jobs), chunk_size)]

        for j, chunk in enumerate(chunks):
            chunk_out = []

            for i, job in enumerate(chunk):
                if 'verbose' in kwargs and kwargs['verbose']:
                    print('\r\r Chunk %d / %d' % (j, nr_chunks) +
                          '\n Working on job %d/%d, ' % (i, len(chunk)) +
                          '\n args: %s, \n kwargs: %s' % (', '.join(job.args), ', '.join([str(tup)
                                                                                     for tup in job.keywords.items()])))
                chunk_out.append(job())

            if chunk_callback is not None:
                ret = chunk_callback(chunk_out, args=[job.args for job in chunk],
                                     kwargs=[job.keywords for job in chunk])
                out.append((chunk_out, ret))
            else:
                out.append((chunk_out, ))

    return list(el[0] for el in zip(*out))

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
