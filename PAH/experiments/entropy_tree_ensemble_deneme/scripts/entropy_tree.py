"""Sabit parametreli entropy-criterion decision tree fabrikasi.

Terminoloji notu: `criterion="entropy"`, Information Gain mantigini kullanan
modern, ikili-bolunmeli (binary-split) bir decision tree'dir -- tarihsel ID3
algoritmasinin kendisi degildir (ID3 cok-yollu bolunme kullanir, budama
icermez). Bu ayrim `03c_ID3_YONTEM_EKLEME_PAH.md`'de zaten netlestirilmisti;
bu deneyde de ayni terminoloji korunur: "Information Gain mantigini kullanan
modern binary-split decision tree", "saf ID3" DENMEZ.

Bu modul hicbir dosyaya yazmaz -- yalnizca bir siniflandirici fabrikasidir.
"""
from sklearn.tree import DecisionTreeClassifier

TREE_PARAMS = dict(
    criterion="entropy",
    max_depth=4,
    min_samples_split=12,
    min_samples_leaf=3,
    class_weight="balanced",
)


def make_tree(random_state):
    """Sabit parametrelerle yeni bir DecisionTreeClassifier orneği doner."""
    return DecisionTreeClassifier(random_state=random_state, **TREE_PARAMS)
