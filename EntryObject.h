#pragma once

#include <Python.h>

#include "Utils.h"


#pragma pack(1)

typedef struct EntryObject {
    PyObject_HEAD

    long descriptor_record;
    int descriptor_word;
    long number;
    int version;
    char source[3 * WORD_LENGTH];
    char line[3 * WORD_LENGTH];
    char telescope[3 * WORD_LENGTH];
    int observation_date;
    int reduction_date;
    float lambda_offset;
    float beta_offset;
    int coordinate_system; // code
    int kind; // code
    int quality; // code
    float position_angle;
    long scan;
    int sub_scan;

    char identifier[WORD_LENGTH];
    int descriptor_version;
    int section_count;
    long entry_length;
    long data_address;
    long data_length;
    long descriptor_number;
    int *section_identifiers;
    long *section_lengths;
    long *section_addresses;
    float *data;
} EntryObject;

#pragma pack()


int Entry_traverse(EntryObject *self,
                   visitproc visit,
                   void *arg);

int Entry_clear(EntryObject *self);

void Entry_dealloc(EntryObject *self);

int Entry_set_source(EntryObject *self,
                     PyObject *value,
                     void *closure);

PyObject *Entry_get_source(const EntryObject *self,
                           void *closure);

int Entry_set_line(EntryObject *self,
                   PyObject *value,
                   void *closure);

PyObject *Entry_get_line(const EntryObject *self,
                         void *closure);


int Entry_set_telescope(EntryObject *self,
                        PyObject *value,
                        void *closure);

PyObject *Entry_get_telescope(const EntryObject *self,
                              void *closure);

Py_ssize_t Entry_length(const EntryObject *self);

PyObject *Entry_item(const EntryObject *self,
                     Py_ssize_t index);

int Entry_ass_item(const EntryObject *self,
                   Py_ssize_t index,
                   PyObject *value);

extern PyMemberDef Entry_members[];

extern PyGetSetDef Entry_getset[];

extern PySequenceMethods Entry_sequence_methods;

extern PyTypeObject EntryType;
