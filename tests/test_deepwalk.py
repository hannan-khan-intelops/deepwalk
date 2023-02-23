#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_deepwalk
----------------------------------

Tests for `deepwalk` module.
"""

import unittest

from deepwalk.skipgram import Skipgram
from deepwalk import walks as serialized_walks
from gensim.models import Word2Vec

from deepwalk import graph
import networkx as nx
import random


class TestDeepwalk(unittest.TestCase):

    def setUp(self):
        self.default_args = {"debug": False, "format": "adjlist", "input": None, "log": "INFO",
                             "matfile_variable_name": "network", "max_memory_data_size": 1000000000, "number_walks": 10,
                             "output": "./deepwalk_output", "representation_size": 64, "seed": 0, "undirected": True,
                             "vertex_freq_degree": False, "walk_length": 40, "window_size": 5, "workers": 1}

        tree_1 = nx.DiGraph()
        tree_1.add_nodes_from([1, 2, 3, 4, 5, 6])
        tree_1.add_edge(1, 2)
        tree_1.add_edge(1, 3)
        tree_1.add_edge(2, 4)
        tree_1.add_edge(2, 5)
        tree_1.add_edge(5, 6)

        G = graph.from_networkx(tree_1, undirected=False)
        print("Number of nodes: {}".format(len(G.nodes())))
        num_walks = len(G.nodes()) * self.default_args["number_walks"]
        print("Number of walks: {}".format(num_walks))
        data_size = num_walks * self.default_args["walk_length"]
        print("Data size (walks*length): {}".format(data_size))
        if data_size < self.default_args["max_memory_data_size"]:
            print("Walking...")
            walks = graph.build_deepwalk_corpus(G, num_paths=self.default_args["number_walks"],
                                                path_length=self.default_args["walk_length"], alpha=0,
                                                rand=random.Random(self.default_args["seed"]))
            print("Training...")
            model = Word2Vec(walks, vector_size=self.default_args["representation_size"],
                             window=self.default_args["window_size"], min_count=0, sg=1,
                             hs=1,
                             workers=self.default_args["workers"])
        else:
            print("Data size {} is larger than limit (max-memory-data-size: {}).  Dumping walks to disk.".format(
                data_size,
                self.default_args["max_memory_data_size"]))
            print("Walking...")

            walks_filebase = self.default_args["output"] + ".walks"
            walk_files = serialized_walks.write_walks_to_disk(G, walks_filebase,
                                                              num_paths=self.default_args["number_walks"],
                                                              path_length=self.default_args["walk_length"], alpha=0,
                                                              rand=random.Random(self.default_args["seed"]),
                                                              num_workers=self.default_args["workers"])

            print("Counting vertex frequency...")
            if not self.default_args["vertex_freq_degree"]:
                vertex_counts = serialized_walks.count_textfiles(walk_files, self.default_args["workers"])
            else:
                # use degree distribution for frequency in tree
                vertex_counts = G.degree(nodes=G.iterkeys())

            print("Training...")
            walks_corpus = serialized_walks.WalksCorpus(walk_files)
            model = Skipgram(sentences=walks_corpus, vocabulary_counts=vertex_counts,
                             size=self.default_args["representation_size"],
                             window=self.default_args["window_size"], min_count=0, trim_rule=None,
                             workers=self.default_args["workers"])

        model.wv.save_word2vec_format(self.default_args["output"])

    def test_something(self):
        # __main__.
        pass

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
