// copyright Luca Istrate, Andrei Medar

#include "./decisionTree.h"  // NOLINT(build/include)
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

using std::string;
using std::pair;
using std::vector;
using std::unordered_map;
using std::make_shared;

// structura unui nod din decision tree
// splitIndex = dimensiunea in functie de care se imparte
// split_value = valoarea in functie de care se imparte
// is_leaf si result sunt pentru cazul in care avem un nod frunza
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
    // TODO(you)
    // Seteaza nodul ca fiind de tip frunza (modificati is_leaf si result)
    // is_single_class = true -> toate testele au aceeasi clasa (acela e result)
    // is_single_class = false -> se alege clasa care apare cel mai des
    if (is_single_class) {
        result = samples[0][0];
    } else {
        vector<int> freq(10);

        for_each(samples.begin(), samples.end(),
                [&freq](const vector<int> &sample) {
            ++freq[sample[0]];
        });

        result = *max_element(freq.begin(), freq.end());
    }

    is_leaf = true;
}

pair<int, int> find_best_split(const vector<vector<int>> &samples,
                               const vector<int> &dimensions) {
    // TODO(you)
    // Intoarce cea mai buna dimensiune si vvector<vector<int>>aloare de split
    // dintre testele
    // primite. Prin cel mai bun split (dimensiune si valoare)
    // ne referim la split-ul care maximizeaza IG
    // pair-ul intors este format din (split_index, split_value)
    int splitIndex = -1, splitValue = -1;
    double maxEntropy = 0.0;
    // std::cerr << "Am initializat" << '\n';

    for_each(dimensions.begin(), dimensions.end(),
            [&splitIndex, &splitValue, &samples, &maxEntropy]
            (const int currDim) {
        // std::cerr << "N-am uinq" << '\n';
        vector<int> uniq = compute_unique(samples, currDim);

        // std::cerr << "Am uinq" << '\n';

        for_each(uniq.begin(), uniq.end(),
                [&splitIndex, &splitValue, &samples, currDim, &maxEntropy]
                (int currVal) {
            pair<vector<vector<int>>, vector<vector<int>>> currSplit
                = split(samples, currDim, currVal);
            if (currSplit.leftSplit.size() && currSplit.rightSplit.size()) {
                // std::cerr << "Am split pt uniq = " << currVal << '\n';
                double currEntropy = get_entropy(samples);
                double leftEntropy = get_entropy(currSplit.leftSplit);
                double rightEntropy = get_entropy(currSplit.rightSplit);

                currEntropy -= (currSplit.first.size() * leftEntropy +
                                currSplit.second.size() * rightEntropy) /
                                samples.size();

                if (currEntropy > maxEntropy) {
                    maxEntropy = currEntropy;
                    splitIndex = currDim;
                    splitValue = currVal;
                }
            }
        });
    });

    return pair<int, int>(splitIndex, splitValue);
}

void Node::train(const vector<vector<int>> &samples) {
    // TODO(you)
    // Antreneaza nodul curent si copii sai, daca e nevoie
    // 1) verifica daca toate testele primite au aceeasi clasa (raspuns)
    // Daca da, acest nod devine frunza, altfel continua algoritmul.
    // 2) Daca nu exista niciun split valid, acest nod devine frunza. Altfel,
    // ia cel mai bun split si continua recursiv
    bool isSingleClass = same_class(samples);
    if (isSingleClass) {
        make_leaf(samples, isSingleClass);
    } else {
        // std::cerr << "Samples: " << samples.size() << '\n';
        // for_each(samples.begin(), samples.end(),
        //         [](const vector<int> &sample) {
        //     for_each(sample.begin(), sample.end(), [](const int sample) {
        //         std::cerr << sample << ' ';
        //     });
        //     std::cerr << '\n';
        // });

        vector<int> randomDim = random_dimensions(785);

        // std::cerr << "Dimensions: " << randomDim.size() << '\n';
        // for_each(randomDim.begin(), randomDim.end(), [](int dim) {
        //     std::cerr << dim << ' ';
        // });
        // std::cerr << "\n";

        pair<int, int> splitParams = find_best_split(samples, randomDim);
        // // std::cerr << "splitIndex: " << splitParams.index
        //           // << "splitValue: " << splitParams.value << '\n';
        //
        if (splitParams.index == -1) {
            make_leaf(samples, isSingleClass);
        } else {
            split_index = splitParams.index;
            split_value = splitParams.value;

            left = make_shared<Node>();
            right = make_shared<Node>();

            pair<vector<vector<int>>, vector<vector<int>>> bestSplit =
                    split(samples, split_index, split_value);

            left->train(bestSplit.leftSplit);
            right->train(bestSplit.rightSplit);
        }
    }
}

int Node::predict(const vector<int> &image) const {
    // TODO(you)
    // Intoarce rezultatul prezis de catre decision tree
    return 0;
}

bool same_class(const vector<vector<int>> &samples) {
    // TODO(you)
    // Verifica daca testele primite ca argument au toate aceeasi
    // clasa(rezultat). Este folosit in train pentru a determina daca
    // mai are rost sa caute split-uri
    int firstClass = samples[0][0];

    auto it = find_if_not(samples.begin(), samples.end(),
            [firstClass](const vector<int> &v) {
        return (v[0] == firstClass);
    });

    return (it == samples.end());
}

float get_entropy(const vector<vector<int>> &samples) {
    // Intoarce entropia testelor primite
    assert(!samples.empty());
    vector<int> indexes;

    int size = samples.size();
    for (int i = 0; i < size; i++) indexes.push_back(i);

    return get_entropy_by_indexes(samples, indexes);
}

float get_entropy_by_indexes(const vector<vector<int>> &samples,
                             const vector<int> &index) {
    // TODO(you)
    // Intoarce entropia subsetului din setul de teste total(samples)
    // Cu conditia ca subsetul sa contina testele ale caror indecsi se gasesc in
    // vectorul index (Se considera doar liniile din vectorul index)
    int numIndexes = index.size();
    float entropy = 0.0;
    vector<float> prob(10);

    for_each(index.begin(), index.end(), [&prob, &samples](const int idx) {
        ++prob[samples[idx][0]];
    });

    for_each(prob.begin(), prob.end(), [numIndexes, &entropy](float &p) {
        if (p) {
            p /= (float)numIndexes;
            entropy -= p * log2(p);
        }
    });

    return entropy;
}

vector<int> compute_unique(const vector<vector<int>> &samples, const int col) {
    // TODO(you)
    // Intoarce toate valorile (se elimina duplicatele)
    // care apar in setul de teste, pe coloana col
    vector<int> uniqueValues;
    vector<bool> found(256);

    // std::cerr << "I'm in; col = " << col << '\n';

    std::for_each(samples.begin(), samples.end(),
            [&found, col](const vector<int> &v) {
        found[v[col]] = true;
    });

    // std::cerr << "Found'em" << '\n';

    for (int i = 0; i < 256; ++i) {
        if (found[i]) {
            uniqueValues.push_back(i);
        }
    }

    // std::cerr << "Put'em" << '\n';

    // sort(uniqueValues.begin(), uniqueValues.end());
    //
    // std::cerr << "Sorted'em" << '\n';

    return uniqueValues;
}

pair<vector<vector<int>>, vector<vector<int>>> split(
    const vector<vector<int>> &samples, const int split_index,
    const int split_value) {
    // Intoarce cele 2 subseturi de teste obtinute in urma separarii
    // In functie de split_index si split_value
    vector<vector<int>> left, right;

    auto p = get_split_as_indexes(samples, split_index, split_value);
    for (const auto &i : p.first) left.push_back(samples[i]);
    for (const auto &i : p.second) right.push_back(samples[i]);

    return pair<vector<vector<int>>, vector<vector<int>>>(left, right);
}

pair<vector<int>, vector<int>> get_split_as_indexes(
    const vector<vector<int>> &samples, const int split_index,
    const int split_value) {
    // TODO(you)
    // Intoarce indecsii sample-urilor din cele 2 subseturi obtinute in urma
    // separarii in functie de split_index si split_value
    vector<int> left, right;
    int len = samples.size();

    for (register int i = 0; i < len; ++i) {
        if (samples[i][split_index] <= split_value) {
            left.push_back(i);
        } else {
            right.push_back(i);
        }
    }

    return make_pair(left, right);
}

vector<int> random_dimensions(const int size) {
    // TODO(you)
    // Intoarce sqrt(size) dimensiuni diferite pe care sa caute splitul maxim
    // Precizare: Dimensiunile gasite sunt > 0 si < size
    vector<int> rez(size);
    int len = (int)sqrt((double)size);

    for (register int i = 0; i < size; ++i) {
        rez[i] = i;
    }

    random_shuffle(rez.begin(), rez.end());
    rez.resize(len);

    return rez;
}
