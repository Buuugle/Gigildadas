#pragma once

#include <python3.13/Python.h>

#include "ContainerObject.h"


typedef struct HeaderObject {
    PyObject_HEAD

    struct EntryHeader *entry_header;

    long number;
    int version;
    PyObject *source;
    PyObject *line;
    PyObject *telescope;
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
} HeaderObject;

int Header_traverse(const HeaderObject *self,
                    visitproc visit,
                    void *arg);

int Header_clear(HeaderObject *self);

void Header_dealloc(HeaderObject *self);

PyObject *Header_new(PyTypeObject *type,
                     PyObject *args,
                     PyObject *kwargs);

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
