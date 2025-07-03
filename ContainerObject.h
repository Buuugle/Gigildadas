#pragma once

#include <python3.13/Python.h>

#include "Utils.h"

#pragma pack(1)

struct FileHeader {
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
};

#pragma pack()


typedef struct ContainerObject {
    PyObject_HEAD

    FILE *input_file;
    FILE *output_file;

    struct FileHeader file_header;
    // TODO: Sections
} ContainerObject;

int Container_traverse(ContainerObject *self,
                       visitproc visit,
                       void *arg);

int Container_clear(ContainerObject *self);

void Container_dealloc(ContainerObject *self);

PyObject *Container_set_input(ContainerObject *self,
                              PyObject *args);

PyObject *Container_get_size(const ContainerObject *self,
                             PyObject *Py_UNUSED(ignored));

PyObject *Container_get_headers(const ContainerObject *self,
                                PyObject *args);

PyObject *Container_read_descriptors(ContainerObject *self,
                                     PyObject *Py_UNUSED(ignored));

extern PyMethodDef Container_methods[];

extern PyTypeObject ContainerType;
