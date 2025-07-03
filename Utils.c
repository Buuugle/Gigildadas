#include "Utils.h"


PyObject *unicode_from_chars(const char *str,
                             const long length) {
    char new_str[length + 1];
    new_str[length] = '\0';
    for (int i = 0; i < length; ++i) {
        new_str[i] = str[i];
    }
    return PyUnicode_FromString(new_str);
}

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
