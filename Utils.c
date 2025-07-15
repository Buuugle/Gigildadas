#include "Utils.h"

double power(const double x,
             const int n) {
    double result = 1.;
    double base = x;
    int exp = n;
    while (exp > 0) {
        if (exp & 1) {
            result *= base;
        }
        base *= base;
        exp >>= 1;
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
