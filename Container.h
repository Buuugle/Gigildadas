#pragma once

#include <Python.h>

#include "Utils.h"

#pragma pack(1)

typedef struct ContainerObject {
    PyObject_HEAD

    FILE *input_file;
    FILE *output_file;

    char file_version[WORD_LENGTH];
    int record_length;
    int file_kind;
    int entry_header_version;
    int entry_header_length;
    int flags;
    long next_entry;
    long next_record;
    int next_word;
    int extension_length_init;
    int extension_count;
    int extension_length_power;
    long *extension_records;
    // TODO: Sections
} ContainerObject;

#pragma pack()

int Container_traverse(ContainerObject *self,
                       visitproc visit,
                       void *arg);

int Container_clear(ContainerObject *self);

void Container_dealloc(ContainerObject *self);

PyObject *Container_set_input(ContainerObject *self,
                              PyObject *args);

PyObject *Container_get_entry_count(const ContainerObject *self,
                                    PyObject *Py_UNUSED(ignored));

PyObject *Container_get_headers(const ContainerObject *self,
                                PyObject *args);

PyObject *Container_get_data(const ContainerObject *self,
                             PyObject *args);

extern PyMethodDef Container_methods[];

extern PyTypeObject ContainerType;
