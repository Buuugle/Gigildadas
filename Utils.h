#pragma once

#include <Python.h>

#define WORD_LENGTH 4

#define MEMORY_ALLOCATION_ERROR PyErr_SetString(PyExc_MemoryError, "failed to allocate memory")
#define FILE_READING_ERROR PyErr_SetString(PyExc_FileError, "failed to read file")
#define DELETE_ERROR PyErr_SetString(PyExc_ValueError, "cannot delete value")

#define INIT_STRING(STR)                    \
    for (int i = 0; i < sizeof(STR); ++i) { \
        STR[i] = ' ';                       \
    }

#define UNICODE_TO_STRING(STR, UNI)                                  \
    if (!UNI) {                                                      \
        DELETE_ERROR;                                                \
        return -1;                                                   \
    }                                                                \
    PyObject *buffer = PyUnicode_AsASCIIString(UNI);                 \
    if (!buffer) {                                                   \
        return -1;                                                   \
    }                                                                \
    const Py_ssize_t arg_length = PyBytes_GET_SIZE(buffer);          \
    const ssize_t length = sizeof(STR);                              \
    if (arg_length > length) {                                       \
        PyErr_SetString(PyExc_ValueError, "wrong string length");    \
        return -1;                                                   \
    }                                                                \
    const char *data = PyBytes_AS_STRING(buffer);                    \
    for (Py_ssize_t i = 0; i < arg_length; ++i) {                    \
        if (data[i] == '\0') {                                       \
            PyErr_SetString(PyExc_ValueError, "null char in value"); \
            return -1;                                               \
        }                                                            \
    }                                                                \
    memcpy(STR, data, arg_length);                                   \
    Py_DECREF(buffer);                                               \
    data = NULL;                                                     \
    buffer = NULL;                                                   \
    for (ssize_t i = arg_length; i < length; ++i) {                  \
        STR[i] = ' ';                                                \
    }                                                                \
    return 0;

#define STRING_TO_UNICODE(STR)           \
    const ssize_t length = sizeof(STR);  \
    char buffer[length + 1];             \
    buffer[length] = '\0';               \
    for (int i = 0; i < length; ++i) {   \
        buffer[i] = STR[i];              \
    }                                    \
    return PyUnicode_FromString(buffer);

#define ARRAY_TO_TUPLE(ARR, LEN, CONV)            \
    PyObject *tuple = PyTuple_New(LEN);           \
    if (!tuple) {                                 \
        MEMORY_ALLOCATION_ERROR;                  \
        return NULL;                              \
    }                                             \
    if (!ARR) {                                   \
        return tuple;                             \
    }                                             \
    for (Py_ssize_t i = 0; i < LEN; ++i) {        \
        PyTuple_SET_ITEM(tuple, i, CONV(ARR[i])); \
    }                                             \
    return tuple;

#define SEQUENCE_TO_ARRAY(ARR, LEN, CONV, SEQ)                             \
    if (!SEQ) {                                                            \
        DELETE_ERROR;                                                      \
        return -1;                                                         \
    }                                                                      \
    PyObject *sequence = PySequence_Fast(SEQ, "value must be a sequence"); \
    if (!sequence) {                                                       \
        return -1;                                                         \
    }                                                                      \
    const Py_ssize_t size = PySequence_Fast_GET_SIZE(sequence);            \
    if (size != LEN) {                                                     \
        Py_DECREF(sequence);                                               \
        PyErr_SetString(PyExc_ValueError, "wrong sequence size");          \
        return -1;                                                         \
    }                                                                      \
    PyObject **items = PySequence_Fast_ITEMS(sequence);                    \
    for (int i = 0; i < LEN; ++i) {                                        \
        const typeof(ARR[i]) temp = CONV(items[i]);                        \
        if (PyErr_Occurred()) {                                            \
            Py_DECREF(sequence);                                           \
            return -1;                                                     \
        }                                                                  \
        ARR[i] = temp;                                                     \
    }                                                                      \
    Py_DECREF(sequence);                                                   \
    return 0;


double power(double x,
             int n);

long max(long a,
         long b);

long min(long a,
         long b);
