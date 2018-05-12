// copyright Luca Istrate, Andrei Medar
#include "randomForest.h"
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
    // TODO(you)
    // Intoarce un vector de marime num_to_return cu elemente random,
    // diferite din samples
    int len = samples.size();
    vector<vector<int>> ret;
    vector<int> pos(len);

    for (register int i = 0; i < len; ++i) {
        pos[i] = i;
    }

    random_shuffle(pos.begin(), pos.end());

    for_each(pos.begin(), pos.begin() + num_to_return, [&samples, &ret](int p) {
        ret.push_back(samples[p]);
    });

    return ret;
}

RandomForest::RandomForest(int num_trees, const vector<vector<int>> &samples)
    : num_trees(num_trees), images(samples) {}

void RandomForest::build() {
    // Aloca pentru fiecare Tree cate n / num_trees
    // Unde n e numarul total de teste de training
    // Apoi antreneaza fiecare tree cu testele alese
    assert(!images.empty());
    vector<vector<int>> random_samples;

    int data_size = images.size() / num_trees;

    // std::cerr << "numTrees = " << num_trees << '\n';

    for (int i = 0; i < num_trees; i++) {
        // cout << "Creating Tree nr: " << i << endl;
        // std::cerr << i << '\n';
        random_samples = get_random_samples(images, data_size);
        // std::cerr << i << '\n';

        // Construieste un Tree nou si il antreneaza
        trees.push_back(Node());
        trees[trees.size() - 1].train(random_samples);
    }
}

int RandomForest::predict(const vector<int> &image) {
    // TODO(you)
    // Va intoarce cea mai probabila prezicere pentru testul din argument
    // se va interoga fiecare Tree si se va considera raspunsul final ca
    // fiind cel majoritar
    vector<int> pred(10);

    for_each(trees.begin(), trees.end(), [&pred, &image](Node &node) {
        ++pred[node.predict(image)];
    });

    return *max_element(pred.begin(), pred.end());
}
