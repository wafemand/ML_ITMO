#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <functional>
#include <algorithm>
#include <queue>
#include <iomanip>

#ifdef LIB

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#endif


using namespace std;


struct Dout {
    Dout() noexcept {
        ios_base::sync_with_stdio(false);
        cin.tie(nullptr);
        cout.tie(nullptr);
#ifdef LOCAL
        freopen("input.txt", "r", stdin);
        freopen("output.txt", "w", stdout);
#else
#ifdef TASK
        freopen(TASK".in", "r", stdin);
        freopen(TASK".out", "w", stdout);
#endif
#endif
    }
} dout;


# define M_PI        3.14159265358979323846    /* pi */
# define M_2_PI        0.63661977236758134308    /* 2/pi */

template<typename T>
T sqr(T a) {
    return a * a;
}

template<typename T>
T cube(T a) {
    return a * a * a;
}


using Vector = vector<double>;
using Table = vector<Vector>;
using VectorInt = vector<int>;
using Dist = function<double(Vector const &, Vector const &)>;
using Kernel = function<double(double)>;


class KNNRegressor {

#define KERNEL(expr) [](double d) { return (expr); }

    map<string, Kernel> kernelMap = {
            {"uniform",      KERNEL(d < 1 ? 0.5 : 0)},
            {"triangular",   KERNEL(d < 1 ? 1 - abs(d) : 0)},
            {"epanechnikov", KERNEL(d < 1 ? 0.75 * (1 - d * d) : 0)},
            {"quartic",      KERNEL(d < 1 ? (15.0 / 16.0) * sqr(1 - d * d) : 0)},
            {"triweight",    KERNEL(d < 1 ? (35.0 / 32.0) * cube(1 - d * d) : 0)},
            {"tricube",      KERNEL(d < 1 ? (70.0 / 81.0) * cube(1 - cube(abs(d))) : 0)},
            {"gaussian",     KERNEL(1 / sqrt(2 * M_PI) * exp(-0.5 * d * d))},
            {"cosine",       KERNEL(d < 1 ? M_PI / 4 * cos(M_PI / 2 * d) : 0)},
            {"logistic",     KERNEL(1 / (2 + exp(d) + exp(-d)))},
            {"sigmoid",      KERNEL(M_2_PI / (exp(d) + exp(-d)))},
            {"zeroWindow",   KERNEL(d <= DBL_EPSILON ? 1 : 0)}
    };

#undef KERNEL

#define DIST [](Vector const &x1, Vector const &x2) -> double
    map<string, Dist> distMap = {
            {"euclidean", DIST {
                double res = 0;
                size_t n = min(x1.size(), x2.size());
                for (size_t i = 0; i < n; i++) {
                    res += sqr(x1[i] - x2[i]);
                }
                return sqrt(res);
            }},
            {"manhattan", DIST {
                double res = 0;
                size_t n = min(x1.size(), x2.size());
                for (size_t i = 0; i < n; i++) {
                    res += abs(x1[i] - x2[i]);
                }
                return res;
            }},
            {"chebyshev", DIST {
                double res = 0;
                size_t n = min(x1.size(), x2.size());
                for (size_t i = 0; i < n; i++) {
                    res = max(res, abs(x1[i] - x2[i]));
                }
                return res;
            }}
    };
#undef DIST

    static double getNthDist(vector<double> const &dists, int k) {
        priority_queue<double> pq;
        for (auto d : dists) {
            pq.push(d);
            if (pq.size() > k) {
                pq.pop();
            }
        }
        return pq.top();
    }

public:
    KNNRegressor(const string &distName, const string &kernelName, bool windowFixed, double windowSize)
            : windowFixed(windowFixed), windowSize(windowSize) {
        auto distIt = distMap.find(distName);
        auto kernelIt = kernelMap.find(kernelName);
        if (distIt == distMap.end()) {
            throw std::invalid_argument("Invalid dist function name: " + distName);
        }
        if (kernelIt == kernelMap.end()) {
            throw std::invalid_argument("Invalid kernel function name: " + kernelName);
        }
        dist = distIt->second;
        kernel = kernelIt->second;
    }

    KNNRegressor() : KNNRegressor("euclidean",
                                  "uniform",
                                  false,
                                  1) {}

    void fit(Table const &X, Table const &y) {
        this->X_ = X;
        this->y_ = y;
    }

    Vector predict(Vector const &x) {
        size_t targetCount = y_[0].size();
        Vector yPred(targetCount);
        Vector dists(X_.size());
        for (size_t i = 0; i < X_.size(); i++) {
            dists[i] = dist(x, X_[i]);
        }
        Kernel curKernel = kernel;
        double h;
        if (windowFixed) {
            h = windowSize;
            if (h < DBL_EPSILON) {
                curKernel = kernelMap["zeroWindow"];
                h = 1;
            }
        } else {
            h = getNthDist(dists, int(windowSize) + 1);
            if (h < DBL_EPSILON) {
                curKernel = kernelMap["zeroWindow"];
                h = 1;
            }
        }
        for (size_t target = 0; target < targetCount; target++) {
            double numer = 0;
            double denom = 0;
            for (size_t i = 0; i < X_.size(); i++) {
                double weight = curKernel(dists[i] / h);
                numer += weight * y_[i][target];
                denom += weight;
            }
            if (denom == 0) {
                for (size_t i = 0; i < X_.size(); i++) {
                    numer += y_[i][target];
                    denom += 1;
                }
            }
            yPred[target] = numer / denom;
        }
        return yPred;
    }


private:
    Table X_;
    Table y_;
    bool windowFixed;
    double windowSize;
    Dist dist;
    Kernel kernel;
};


#ifndef LIB

void cf_main() {
    int n, m;
    cin >> n >> m;
    Table X;
    Table y(n);
    for (int i = 0; i < n; i++) {
        Vector features(m);
        for (int j = 0; j < m; j++) {
            cin >> features[j];
        }
        y[i].resize(1);
        cin >> y[i][0];
        X.emplace_back(move(features));
    }
    Vector query(m);
    for (int i = 0; i < m; i++) {
        cin >> query[i];
    }
    string dist;
    string kernel;
    string windowType;
    cin >> dist >> kernel >> windowType;
    int window;
    cin >> window;
    KNNRegressor knn(dist, kernel, windowType == "fixed", window);
    knn.fit(X, y);
    cout << fixed << setprecision(10) << knn.predict(query)[0];
}

void print_usage() {
    cerr << "KNNRegressor <window size> <kernel type> <dist type> <window type>" << endl;
}

int main(int argc, char **argv) {
    if (argc == 1) {
        cf_main();
        return 0;
    }

    if (argc != 5) {
        print_usage();
        return EXIT_FAILURE;
    }

    double windowSize = stod(argv[1]);
    string distType = argv[2];
    string kernelType = argv[3];
    string windowType = argv[4];

    Table X, y;
    int featuresCount, targetsCount;

    KNNRegressor knnRegressor(distType, kernelType, windowType == "fixed", windowSize);

    while (true) {
        string command;
        cin >> command;
        cerr << command << endl;
        if (command == "exit") {
            break;
        } else if (command == "sizes") {
            cin >> featuresCount >> targetsCount;
        } else if (command == "add") {
            Vector rowX(featuresCount);
            Vector rowy(targetsCount);
            for (int i = 0; i < featuresCount; i++) {
                cin >> rowX[i];
            }
            for (int i = 0; i < targetsCount; i++) {
                cin >> rowy[i];
            }
            X.push_back(rowX);
            y.push_back(rowy);
        } else if (command == "reset") {
            X.clear();
            y.clear();
        } else if (command == "predict") {
            knnRegressor.fit(X, y);
            Vector x(featuresCount);
            for (int i = 0; i < featuresCount; i++) {
                cin >> x[i];
            }
            auto yPred = knnRegressor.predict(x);
            for (int i = 0; i < targetsCount; i++) {
                cout << yPred[i] << ' ';
            }
        }
    }

}

#else

namespace py = pybind11;

//PYBIND11_MAKE_OPAQUE(Vector)
//PYBIND11_MAKE_OPAQUE(Table)

PYBIND11_MODULE(KNNRegressor, m) {
    py::class_<KNNRegressor>(m, "KNNRegressor")
            .def(
                    py::init([](
                            double windowSize,
                            const string &distName,
                            const string &kernelName,
                            const string &windowType
                    ) {
                        return make_unique<KNNRegressor>(
                                distName,
                                kernelName,
                                windowType == "fixed",
                                windowSize
                        );
                    }),
                    py::arg("windowSize"),
                    py::arg("distName") = "euclidean",
                    py::arg("kernelName") = "uniform",
                    py::arg("windowType") = "fixed"
            )
            .def(
                    "fit",
                    &KNNRegressor::fit
            )
            .def(
                    "predict",
                    &KNNRegressor::predict
            );

    m.def("kek", [](){
        return Vector(10, 10);
    });
    m.def("lol", [](const Table &table){
        return table[0];
    });

    m.doc() = "KNN binding"; // optional module docstring
}

#endif
