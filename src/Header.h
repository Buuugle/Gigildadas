#pragma once

#include <stdint.h>
#include <Python.h>

#include "Utils.h"


#pragma pack(1)

typedef struct HeaderObject {
    PyObject_HEAD

    int64_t descriptor_record;
    int32_t descriptor_word;
    int64_t number;
    int32_t version;
    char source[3 * WORD_SIZE];
    char line[3 * WORD_SIZE];
    char telescope[3 * WORD_SIZE];
    int32_t observation_date;
    int32_t reduction_date;
    float lambda_offset;
    float beta_offset;
    int32_t coordinate_system; // code
    int32_t kind; // code
    int32_t quality; // code
    float position_angle;
    int64_t scan;
    int32_t sub_scan;

    char identifier[WORD_SIZE];
    int32_t descriptor_version;
    int32_t section_count;
    int64_t entry_length;
    int64_t data_address;
    int64_t data_size;
    int64_t descriptor_number;
    int32_t *section_identifiers;
    int64_t *section_lengths;
    int64_t *section_addresses;
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
