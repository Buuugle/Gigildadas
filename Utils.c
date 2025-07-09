#include "Utils.h"

double power(const double x,
             const int n) {
    double result = 1.;
    for (int i = 0; i < n; ++i) {
        result *= x;
    }
    return result;
}

long max(const long a,
         const long b) {
    return a > b ? a : b;
}

long min(const long a,
         const long b) {
    return a < b ? a : b;
}
