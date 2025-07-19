#pragma once

#include <stdint.h>
#include <Python.h>

#include "Utils.h"

#pragma pack(1)

typedef struct ContainerObject {
    PyObject_HEAD

    FILE *input_file;
    FILE *output_file;

    char file_version[WORD_SIZE];
    int32_t record_length;
    int32_t file_kind;
    int32_t entry_header_version;
    int32_t entry_header_length;
    int32_t flags;
    int64_t next_entry;
    int64_t next_record;
    int32_t next_word;
    int32_t extension_length_init;
    int32_t extension_count;
    int32_t extension_length_power;
    int64_t *extension_records;
} ContainerObject;

#pragma pack()

int Container_traverse(ContainerObject *self,
                       visitproc visit,
                       void *arg);

int Container_clear(ContainerObject *self);

void Container_dealloc(ContainerObject *self);

PyObject *Container_set_input(ContainerObject *self,
                              PyObject *args,
                              PyObject *kwargs);

PyObject *Container_get_size(const ContainerObject *self,
                             PyObject *Py_UNUSED(ignored));

PyObject *Container_get_headers(const ContainerObject *self,
                                PyObject *args,
                                PyObject *kwargs);

PyObject *Container_get_data(const ContainerObject *self,
                             PyObject *args,
                             PyObject *kwargs);

PyObject *Container_get_sections(const ContainerObject *self,
                                 PyObject *args,
                                 PyObject *kwargs);

extern PyMethodDef Container_methods[];

extern PyTypeObject ContainerType;
