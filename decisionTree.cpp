#include "./decisionTree.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>
#include <string>

#define leftSplit first
#define rightSplit second
#define index first
#define value second
#define UCHAR_MAX 256

using std::string;
using std::pair;
using std::vector;
using std::unordered_map;
using std::make_shared;

Node::Node() {
    is_leaf = false;
    left = nullptr;
    right = nullptr;
}

void Node::make_decision_node(const int index, const int val) {
    split_index = index;
    split_value = val;
}

void Node::make_leaf(const vector<vector<int>> &samples,
                     const bool is_single_class) {
    if (is_single_class) {
        // daca samples reprezinta o singura cifra, result devine aceasta cifra
        result = samples[0][0];
    } else {
        vector<int> freq(10);

        // se calculeaza numarul de aparitii al fiecarei cifre din samples
        for_each(samples.begin(), samples.end(),
                [&freq](const vector<int> &sample) {
            ++freq[sample[0]];
        });

        // se determina cifra ce apare de cele mai multe ori, dupa care aceasta
        // se retine in result
        vector<int>::iterator itToMax = max_element(freq.begin(), freq.end());
        result = distance(freq.begin(), itToMax);
    }

    is_leaf = true;
}

pair<int, int> find_best_split(const vector<vector<int>> &samples,
                               const vector<int> &dimensions) {
    int splitIndex = -1, splitValue = -1;
    float maxGain = 0.0;

    // se calculeaza setului de date curent
    float currEntropy = get_entropy(samples);

    // se parcurg dimensions, pentru fiecare element din acestea se genereaza
    // vectorul uniq ce contine valorile unicizate de pe coloana elementului
    // respectiv;
    // split_index si split_value se vor selecta dintre elementele din
    // dimensions, respectiv dintre valorile din vectorul uniq corespunzator
    // fiecarei dimensiuni
    for_each(dimensions.begin(), dimensions.end(),
            [&splitIndex, &splitValue, &samples, &maxGain, currEntropy]
            (const int currDim) {
        vector<int> uniq = compute_unique(samples, currDim);

        for_each(uniq.begin(), uniq.end(),
                [&splitIndex, &splitValue, &samples,
                currDim, &maxGain, currEntropy](int currVal) {
            // se genereaza split-ul sub forma de indecsi
            auto currSplit = get_split_as_indexes(samples, currDim, currVal);

            // splitul este valid, se calculeaza entropiile pentru split-urile
            // stang si drept, precum si inofrmation gain-ul corespunzator
            // acestui split
            if (currSplit.leftSplit.size() && currSplit.rightSplit.size()) {
                float leftEntropy =
                    get_entropy_by_indexes(samples, currSplit.leftSplit);
                float rightEntropy =
                    get_entropy_by_indexes(samples, currSplit.rightSplit);

                float infoGain = currEntropy -
                                 (currSplit.leftSplit.size() * leftEntropy +
                                 currSplit.rightSplit.size() * rightEntropy) /
                                 samples.size();

                // daca s-a gasit un split mai bun decat cele precedente,
                // se modifica parametrii nodului
                if (infoGain > maxGain) {
                    maxGain = infoGain;
                    splitIndex = currDim;
                    splitValue = currVal;
                }
            }
        });
    });

    return pair<int, int>(splitIndex, splitValue);
}

void Node::train(const vector<vector<int>> &samples) {
    if (same_class(samples)) {
        // daca samples reprezinta toate aceeasi cifra, se creeaza o frunza
        make_leaf(samples, true);
    } else {
        int len = samples[0].size();
        auto splitParams = find_best_split(samples, random_dimensions(len));

        if (splitParams.index == -1) {
            // daca toate split-urile sunt nule, se creeaza tot o frunza
            make_leaf(samples, false);
        } else {
            // se creeaza un nod de decizie cu split_index si split_value
            // returnate de functia find_best_split;
            make_decision_node(splitParams.index - 1, splitParams.value);

            // se continua cu antrenarea nodurilor stang si drept, in functie
            // de split-ul optim gasit
            left = make_shared<Node>();
            right = make_shared<Node>();

            auto bestSplit = split(samples, split_index + 1, split_value);

            right->train(bestSplit.rightSplit);
            left->train(bestSplit.leftSplit);
        }
    }
}

int Node::predict(const vector<int> &image) const {
    // se cauta cifra din image, pe principiul unui arbore binar de cautare
    if (is_leaf) {
        return result;
    }

    if (image[split_index] <= split_value) {
        return left->predict(image);
    }

    return right->predict(image);
}

bool same_class(const vector<vector<int>> &samples) {
    // se ia ca etalon clasa primului sample
    int firstClass = samples[0][0];

    // se cauta un sample ce apartine altei clase decat cea luata drept etalon
    auto diffClass = find_if_not(samples.begin(), samples.end(),
            [firstClass](const vector<int> &v) {
        return (v[0] == firstClass);
    });

    return (diffClass == samples.end());
}

float get_entropy(const vector<vector<int>> &samples) {
    assert(!samples.empty());
    vector<int> indexes;

    int size = samples.size();
    for (int i = 0; i < size; i++) indexes.push_back(i);

    return get_entropy_by_indexes(samples, indexes);
}

float get_entropy_by_indexes(const vector<vector<int>> &samples,
                             const vector<int> &index) {
    float numIndexes = index.size();
    float entropy = 0.0;
    vector<float> prob(10);

    // se calculeaza frecventa aparitiei fiecarei clase printre liniile din
    // samples date de fiecare index
    for_each(index.begin(), index.end(), [&prob, &samples](const int idx) {
        ++prob[samples[idx][0]];
    });

    // se calculeaza efectiv entropia
    for_each(prob.begin(), prob.end(), [numIndexes, &entropy](float p) {
        if (p) {
            p /= (float)numIndexes;
            entropy -= p * log2(p);
        }
    });

    return entropy;
}

vector<int> compute_unique(const vector<vector<int>> &samples, const int col) {
    vector<int> uniqueValues;
    vector<bool> found(UCHAR_MAX);

    // se completeaza vectorul de aparitii corespunzator valorilor fiecarui
    // pixel de pe coloana col din samples
    for_each(samples.begin(), samples.end(),
            [&found, col](const vector<int> &sample) {
        found[sample[col]] = true;
    });

    // se adauga in vector pixelii gasiti
    for (int i = 0; i < UCHAR_MAX; ++i) {
        if (found[i]) {
            uniqueValues.push_back(i);
        }
    }

    return uniqueValues;
}

pair<vector<vector<int>>, vector<vector<int>>> split(
        const vector<vector<int>> &samples, const int split_index,
        const int split_value) {
    vector<vector<int>> left, right;

    auto p = get_split_as_indexes(samples, split_index, split_value);
    for (const auto &i : p.first) left.push_back(samples[i]);
    for (const auto &i : p.second) right.push_back(samples[i]);

    return pair<vector<vector<int>>, vector<vector<int>>>(left, right);
}

pair<vector<int>, vector<int>> get_split_as_indexes(
        const vector<vector<int>> &samples, const int split_index,
        const int split_value) {
    vector<int> leftSample, rightSample;
    int len = samples.size();

    // se creeaza sample-urile stang si drept, in functie de split_index si
    // split_value
    for (int i = 0; i < len; ++i) {
        if (samples[i][split_index] <= split_value) {
            leftSample.push_back(i);
        } else {
            rightSample.push_back(i);
        }
    }

    return make_pair(leftSample, rightSample);
}

vector<int> random_dimensions(const int size) {
    int len = (int)sqrt((double)size);
    vector<int> rez(len);
    vector<bool> found(size, false);

    // se foloseste mersenne twister-ul pentru a genera dimensiuni distribuite
    // uniform in intervalul [1, size - 1]
    std::random_device rd;
    std::mt19937 engine(rd());
    std::uniform_int_distribution<int> dist(1, size - 1);

    // se genereaza cele sqrt(size) dimensiuni dorite
    for (register int i = 0; i < len;) {
        int x = dist(engine);

        if (!found[x]) {
            rez[i++] = x;
            found[x] = true;
        }
    }

    return rez;
}
=======
// MLC
// PUTU - 2011
>>>>>>> 0bde432e40303b0f7fff31680ceb0a4ac3ae4b1f
