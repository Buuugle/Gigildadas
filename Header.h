#pragma once

#include <Python.h>

#include "Utils.h"


#pragma pack(1)

typedef struct HeaderObject {
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
} HeaderObject;

#pragma pack()


int Header_traverse(HeaderObject *self,
                    visitproc visit,
                    void *arg);

int Header_clear(HeaderObject *self);

void Header_dealloc(HeaderObject *self);

PyObject *Header_new(PyTypeObject *type,
                     PyObject *arg,
                     PyObject *kwds);

int Header_set_source(HeaderObject *self,
                      PyObject *value,
                      void *closure);

PyObject *Header_get_source(const HeaderObject *self,
                            void *closure);

int Header_set_line(HeaderObject *self,
                    PyObject *value,
                    void *closure);

PyObject *Header_get_line(const HeaderObject *self,
                          void *closure);


int Header_set_telescope(HeaderObject *self,
                         PyObject *value,
                         void *closure);

PyObject *Header_get_telescope(const HeaderObject *self,
                               void *closure);

extern PyMemberDef Header_members[];

extern PyGetSetDef Header_getset[];

extern PyTypeObject HeaderType;
