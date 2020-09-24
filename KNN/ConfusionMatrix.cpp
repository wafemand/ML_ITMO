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

#endif


using namespace std;

#ifndef LIB

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

#endif

class ConfusionMatrix {
private:
    vector<vector<int> > matrix;
    size_t n;
    vector<int> predictedClassSizes;
    vector<int> realClassSizes;
    int all;

    static double F(double prec, double rec) {
        if (prec <= DBL_EPSILON || rec <= DBL_EPSILON) {
            return 0;
        }
        //return 2 / (1 / prec + 1 / rec);
        return 2 * prec * rec / (prec + rec);
    }

public:
    explicit ConfusionMatrix(vector<vector<int> > const &matrix)
            : matrix(matrix), n(matrix.size()),
              predictedClassSizes(n),
              realClassSizes(n),
              all(0) {
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++) {
                predictedClassSizes[i] += matrix[i][j];
                realClassSizes[j] += matrix[i][j];
                all += matrix[i][j];
            }
        }
    }

    double precision(size_t classIndex) {
        if (matrix[classIndex][classIndex] == 0 &&
            matrix[classIndex][classIndex] == predictedClassSizes[classIndex]) {
            return 0;
        }
        return double(matrix[classIndex][classIndex]) / predictedClassSizes[classIndex];
    }

    double precision() {
        double res = 0;
        for (size_t i = 0; i < n; i++) {
            res += realClassSizes[i] * precision(i);
        }
        return res / all;
    }

    double recall(size_t classIndex) {
        if (matrix[classIndex][classIndex] == 0 &&
            matrix[classIndex][classIndex] == realClassSizes[classIndex]) {
            return 0;
        }
        return double(matrix[classIndex][classIndex]) / realClassSizes[classIndex];
    }

    double recall() {
        double res = 0;
        for (size_t i = 0; i < n; i++) {
            res += matrix[i][i];
        }
        return res / all;
    }

    double class_F(size_t classIndex) {
        return F(precision(classIndex), recall(classIndex));
    }

    double micro_F() {
        double res = 0;
        for (size_t i = 0; i < n; i++) {
            res += realClassSizes[i] * class_F(i);
        }
        return res / all;
    }

    double macro_F() {
        return F(precision(), recall());
    }
};


#ifndef LIB


int main() {
    int n;
    cin >> n;
    vector<vector<int> > cm(n, vector<int>(n));
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            cin >> cm[j][i];
        }
    }

    ConfusionMatrix confusionMatrix(cm);
    cout << fixed << setprecision(10) << confusionMatrix.macro_F() << endl;
    cout << fixed << setprecision(10) << confusionMatrix.micro_F() << endl;
}

#else

namespace py = pybind11;

PYBIND11_MODULE(ConfusionMatrix, m) {
    py::class_<ConfusionMatrix>(m, "ConfusionMatrix")
            .def(py::init([](const vector<vector<int>> &matrix) {
                return make_unique<ConfusionMatrix>(matrix);
            }))
            .def("f1_score_micro", &ConfusionMatrix::micro_F)
            .def("f1_score_macro", &ConfusionMatrix::macro_F);
}

#endif