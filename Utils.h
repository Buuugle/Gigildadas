#pragma once

#include <python3.13/Python.h>

#define WORD_LENGTH 4


PyObject *unicode_from_chars(const char *str,
                             long length);

double power(double x,
             int n);

long max(long a,
         long b);

long min(long a,
         long b);
