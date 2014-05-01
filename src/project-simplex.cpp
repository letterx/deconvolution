#include "project-simplex.hpp"

#include <algorithm>

void projectSimplex(const std::vector<double>& x, std::vector<double>& result) {
    result = x;
    std::sort(result.begin(), result.end());

    const int n = x.size();
    int i = n-1;
    double t_i;
    while (i >= 0) {
        t_i = 0;
        for (int j = i+1; j <= n; ++j)
            t_i += result[j-1];
        t_i = (t_i - 1.0) / (n - i);
        if (t_i >= result[i-1])
            break;
        i--;
    }

    for (int i = 0; i < n; ++i) 
        result[i] = std::max(x[i] - t_i, 0.0);

}
