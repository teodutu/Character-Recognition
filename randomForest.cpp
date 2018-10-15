// copyright Luca Istrate, Andrei Medar
// Copyright Dutu Teodor-Stefan, Popescu Daniel-Octavian

#include "./randomForest.h"
#include <iostream>
#include <random>
#include <vector>
#include <string>
#include "decisionTree.h"

using std::vector;
using std::pair;
using std::string;
using std::mt19937;

vector<vector<int>> get_random_samples(const vector<vector<int>> &samples,
                                       int num_to_return) {
    int len = samples.size();
    vector<vector<int>> randomSamples;
    vector<int> pos(num_to_return);
    vector<bool> found(len);

    // se foloseste mersenne twister-ul pentru a genera indici pentru samples
    // distribuiti uniform in intervalul [0, samples.size() - 1]
    std::random_device rd;
    std::mt19937 engine(rd());
    std::uniform_int_distribution<int> dist(0, len - 1);

    // se genereaza mai intai indicii celor num_to_return sample-uri dorite
    for (register int i = 0; i < num_to_return;) {
        int x = dist(engine);

        if (!found[x]) {
            pos[i++] = x;
            found[x] = true;
        }
    }

    // se introduc in vectorul randomSamples liniile sample-urile ale caror
    // indici au fost generate anterior
    for_each(pos.begin(), pos.end(), [&samples, &randomSamples](int p) {
        randomSamples.push_back(samples[p]);
    });

    return randomSamples;
}

RandomForest::RandomForest(int num_trees, const vector<vector<int>> &samples)
    : num_trees(num_trees), images(samples) {}

void RandomForest::build() {
    assert(!images.empty());
    vector<vector<int>> random_samples;

    int data_size = images.size() / num_trees;

    for (int i = 0; i < num_trees; i++) {
        random_samples = get_random_samples(images, data_size);

        trees.push_back(Node());
        trees[trees.size() - 1].train(random_samples);
    }
}

int RandomForest::predict(const vector<int> &image) {
    vector<int> pred(10);

    // se contorizeaza aparitiile fiecarei clase
    for_each(trees.begin(), trees.end(), [&pred, &image] (Node &tree) {
        ++pred[tree.predict(image)];
    });

    // se returneaza clasa ce apare de cele mai multe ori
    auto pos = max_element(pred.begin(), pred.end());
    return distance(pred.begin(), pos);
}
