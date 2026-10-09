"""21-agac, dengeli-bootstrap, hard-voting entropy-tree ensemble.

Her agac icin:
  1. Dengeli (balanced) bootstrap: pozitif ve negatif siniflardan AYRI AYRI,
     yerine-koyarak (with replacement), her sinifin orijinal egitim
     buyuklugu kadar orneklem cekilir, sonra birlestirilir. Bu, sinif
     agirligi (`class_weight="balanced"`) ile AYRI, ek bir dengesizlik-
     duzeltmesi katmanidir -- "dengeli bootstrap" ifadesi bunu karsilar.
  2. (Yalnizca Model C icin) `n_features_per_tree` verilirse, agac o
     alt-kumeden rastgele secilen `n_features_per_tree` kolonla fit edilir
     (ozellik-bagimsiz agac cesitliligi).

Hard voting: her agacin 0/1 tahmini toplanir, oy sayisi >= esik ise sinif 1.

Bu modul hicbir dosyaya yazmaz -- yalnizca model/tahmin nesneleridir.
"""
import numpy as np

from entropy_tree import make_tree

N_TREES = 21


def _balanced_bootstrap_indices(y, rng):
    """Pozitif/negatif siniflardan ayri ayri, kendi orijinal buyuklukleriyle,
    yerine koyarak orneklenmis indeksleri dondurur (karistirilmis).
    """
    y = np.asarray(y)
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    sampled_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
    sampled_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
    combined = np.concatenate([sampled_pos, sampled_neg])
    rng.shuffle(combined)
    return combined


class EntropyTreeEnsemble:
    """21 agaclik, dengeli-bootstrap, hard-voting ensemble.

    `n_features_per_tree=None` ise (Model A/B/D) her agac TUM verilen
    kolonlarla fit edilir. Bir tamsayi verilirse (Model C) her agac o
    sayida rastgele kolonla fit edilir -- hangi kolonlarin secildigi
    `self.tree_feature_subsets_` icinde saklanir (predict icin gerekli).
    """

    def __init__(self, base_seed, n_trees=N_TREES, n_features_per_tree=None):
        self.base_seed = base_seed
        self.n_trees = n_trees
        self.n_features_per_tree = n_features_per_tree

    def fit(self, X, y):
        self.feature_names_ = list(X.columns)
        self.trees_ = []
        self.tree_feature_subsets_ = []
        for k in range(self.n_trees):
            # her agac icin ayri, deterministik tekil seed: base_seed'den
            # turetilir (03c'nin `SEED + repeat_idx*100 + fold_idx` seed
            # formuluyle tutarli -- base_seed zaten bu formulle uretiliyor,
            # burada agac-indeksine gore ek bir ofset ekleniyor).
            tree_seed = self.base_seed * 1000 + k
            rng = np.random.RandomState(tree_seed)

            if self.n_features_per_tree is not None:
                cols = list(rng.choice(self.feature_names_, size=self.n_features_per_tree, replace=False))
            else:
                cols = self.feature_names_
            self.tree_feature_subsets_.append(cols)

            boot_idx = _balanced_bootstrap_indices(y.values if hasattr(y, "values") else y, rng)
            X_boot = X[cols].iloc[boot_idx]
            y_boot = y.iloc[boot_idx] if hasattr(y, "iloc") else y[boot_idx]

            tree = make_tree(random_state=tree_seed)
            tree.fit(X_boot, y_boot)
            self.trees_.append(tree)
        return self

    def vote_counts(self, X):
        """Her satir icin, 21 agactan kac tanesinin sinif=1 dedigini dondurur."""
        votes = np.zeros(len(X), dtype=int)
        for tree, cols in zip(self.trees_, self.tree_feature_subsets_):
            votes += tree.predict(X[cols]).astype(int)
        return votes

    def predict(self, X, threshold):
        """`threshold` oydan az OLMAYAN (>=) satirlar icin sinif=1 doner."""
        return (self.vote_counts(X) >= threshold).astype(int)
