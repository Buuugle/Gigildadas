#pragma once

#include <Python.h>

#define WORD_LENGTH 4

#define UNICODE_TO_CHARS(STR, UNICODE)                                  \
    if (!UNICODE) {                                                     \
        PyErr_SetString(PyExc_TypeError, "cannot delete object");       \
        return -1;                                                      \
    }                                                                   \
    if (!PyUnicode_Check(UNICODE)) {                                    \
        PyErr_SetString(PyExc_TypeError, "value must be a string");     \
        return -1;                                                      \
    }                                                                   \
    const ssize_t length = sizeof(STR);                                 \
    Py_ssize_t arg_length;                                              \
    wchar_t *buffer = PyUnicode_AsWideCharString(UNICODE, &arg_length); \
    if (!buffer) {                                                      \
        return -1;                                                      \
    }                                                                   \
    if (arg_length < length) {                                          \
        PyErr_SetString(PyExc_ValueError, "wrong string length");       \
        return -1;                                                      \
    }                                                                   \
    wcstombs(STR, buffer, length);                                      \
    PyMem_Free(buffer);                                                 \
    return 0;

#define CHARS_TO_UNICODE(STR)            \
    const ssize_t length = sizeof(STR);  \
    char buffer[length + 1];             \
    buffer[length] = '\0';               \
    for (int i = 0; i < length; ++i) {   \
        buffer[i] = STR[i];              \
    }                                    \
    return PyUnicode_FromString(buffer);

#define MEMORY_ALLOCATION_ERROR PyErr_SetString(PyExc_MemoryError, "failed to allocate memory")
#define FILE_READING_ERROR PyErr_SetString(PyExc_FileError, "failed to read file")


double power(double x,
             int n);

long max(long a,
         long b);

long min(long a,
         long b);
