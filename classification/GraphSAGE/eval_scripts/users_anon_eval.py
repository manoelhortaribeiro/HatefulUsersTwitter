from __future__ import print_function
import json
import numpy as np

from networkx.readwrite import json_graph
from argparse import ArgumentParser


def run_regression(train_embeds, train_labels, test_embeds, test_labels):
    np.random.seed(1)
    from sklearn.linear_model import SGDClassifier
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import f1_score
    dummy = DummyClassifier()
    dummy.fit(train_embeds, train_labels)
    log = SGDClassifier(loss="log", n_jobs=55)
    log.fit(train_embeds, train_labels)
    print("Test scores")
    print(f1_score(test_labels, log.predict(test_embeds), average="micro"))
    print("Train scores")
    print(f1_score(train_labels, log.predict(train_embeds), average="micro"))
    print("Random baseline")
    print(f1_score(test_labels, dummy.predict(test_embeds), average="micro"))


if __name__ == '__main__':
    parser = ArgumentParser("Run evaluation on Hate Speech data.")
    parser.add_argument("dataset_dir", help="Path to directory containing the dataset.")
    parser.add_argument("embed_dir",
                        help="Path to directory containing the learned node embeddings. Set to 'feat' for raw features.")
    parser.add_argument("setting", help="Either val or test.")
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    data_dir = args.embed_dir
    setting = args.setting

    print("Loading data...")
    G = json_graph.node_link_graph(json.load(open(dataset_dir + "/users_anon-G.json")))
    labels = json.load(open(dataset_dir + "/users_anon-class_map.json"))

    train_ids = [n for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']]
    test_ids = [n for n in G.nodes() if G.node[n][setting]]
    train_labels = [labels[i] for i in train_ids]
    test_labels = [labels[i] for i in test_ids]

    if data_dir == "feat":
        print("Using only features..")
        feats = np.load(dataset_dir + "/users_anon-feats.npy")

        ## Logistic gets thrown off by big counts, so log transform num comments and score
        for i in [0, 1, 2, 3, 4, 5, 6]:
            feats[:, i] = np.log(feats[:, i] + 1.0)
        feat_id_map = json.load(open(dataset_dir + "users_anon-id_map.json"))
        feat_id_map = {id: val for id, val in feat_id_map.iteritems()}
        train_feats = feats[[feat_id_map[id] for id in train_ids]]
        test_feats = feats[[feat_id_map[id] for id in test_ids]]
        print("Running regression..")
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit(train_feats)
        train_feats = scaler.transform(train_feats)
        test_feats = scaler.transform(test_feats)
        run_regression(train_feats, train_labels, test_feats, test_labels)