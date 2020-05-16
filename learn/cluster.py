import numpy as np
from functools import partial
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

try:
    from learn.utils import train_predict_conf, contingency_distance, simulated_annealing, validate
    from preproc.transformers import FunctionPredictor
except ModuleNotFoundError:
    import seminar
    from seminar.learn.utils import train_predict_conf, contingency_distance, simulated_annealing, validate
    from seminar.preproc.transformers import FunctionPredictor


def cluster_based_classif(samples, gt_samples, pip, n_clusters, score, nan_val=-1, freq_weighted=True, train=True,
                          annealing_params={}):
    print('TRAIN')
    if train:
        _ = pip.fit(samples)

    clf = pip.predict(samples)
    valids, conf = validate(clf.flatten(), gt_samples.flatten())

    gtr_labels = np.unique(gt_samples[valids])
    cluster_labels = np.unique(clf[valids])
    labels = np.r_[cluster_labels, gtr_labels]

    max_assignment = labels[np.argmax(conf, axis=0)][:n_clusters]

    label_set, label_counts = np.unique(gt_samples, return_counts=True)

    # take out -1
    valid_idx = np.where(label_set != nan_val)
    label_set = label_set[valid_idx]
    label_counts = label_counts[valid_idx]

    if freq_weighted:
        argsort = np.argsort(label_set)
        label_set = label_set[argsort]
        label_counts = label_counts[argsort]

        label_weights = label_counts / np.sum(label_counts)
    else:
        label_weights = None

    print('MAP')
    calc_contingency_dist = partial(contingency_distance, score=partial(score, average=None),
                                    ground_truth=gt_samples.flatten(), clustering=clf.flatten(),)
                                    #trn_inds=gt_samples.index)

    x0 = max_assignment[:n_clusters].copy()
    assign, hist, T_end, samples = simulated_annealing(x0, calc_contingency_dist,
                                                       label_set=label_set, label_weights=label_weights,
                                                       **annealing_params)

    assignment_to_label = dict(zip(range(len(assign)), assign))
    assignment_to_label[nan_val] = nan_val

    pip = Pipeline([('pip', pip),
                    ('map', FunctionPredictor(lambda cluster: assignment_to_label.get(cluster, None)))
                    ])

    return pip, assign, dict(hist=hist, T_end=T_end, samples=samples)


