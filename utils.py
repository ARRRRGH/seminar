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
