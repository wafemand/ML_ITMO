#include <bits/stdc++.h>

using namespace std;

int main() {
    int n, m, k;
    cin >> n >> m >> k;
    vector<pair<int, int> > c(n);
    for (int i = 0; i < n; i++) {
        cin >> c[i].first;
        c[i].second = i + 1;
    }
    sort(c.begin(), c.end());
    for (int i = 0; i < k; i++) {
        cout << (n - i + k - 1) / k << ' ';
        for (int j = i; j < n; j += k) {
            cout << c[j].second << ' ';
        }
        cout << '\n';
    }
}

