#include <stddef.h>

#include "Header.h"


int Header_traverse(HeaderObject *self,
                    visitproc visit,
                    void *arg) {
    return 0;
}

int Header_clear(HeaderObject *self) {
    PyMem_Free(self->section_identifiers);
    PyMem_Free(self->section_lengths);
    PyMem_Free(self->section_addresses);
    self->section_identifiers = NULL;
    self->section_lengths = NULL;
    self->section_addresses = NULL;
    return 0;
}

void Header_dealloc(HeaderObject *self) {
    PyObject_GC_UnTrack(self);
    Header_clear(self);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

int Header_set_source(HeaderObject *self,
                      PyObject *value,
                      void *closure) {
    UNICODE_TO_CHARS(self->source, value)
}

PyObject *Header_get_source(const HeaderObject *self,
                            void *closure) {
    CHARS_TO_UNICODE(self->source)
}

int Header_set_line(HeaderObject *self,
                    PyObject *value,
                    void *closure) {
    UNICODE_TO_CHARS(self->line, value)
}

PyObject *Header_get_line(const HeaderObject *self,
                          void *closure) {
    CHARS_TO_UNICODE(self->line)
}

int Header_set_telescope(HeaderObject *self,
                         PyObject *value,
                         void *closure) {
    UNICODE_TO_CHARS(self->telescope, value)
}

PyObject *Header_get_telescope(const HeaderObject *self,
                               void *closure) {
    CHARS_TO_UNICODE(self->telescope)
}

PyMemberDef GeneralSection_members[] = {
    {
        .name = "number",
        .type = Py_T_LONG,
        .offset = offsetof(HeaderObject, number)
    },
    {
        .name = "version",
        .type = Py_T_INT,
        .offset = offsetof(HeaderObject, version)
    },
    {
        .name = "observation_date",
        .type = Py_T_INT,
        .offset = offsetof(HeaderObject, observation_date)
    },
    {
        .name = "reduction_date",
        .type = Py_T_INT,
        .offset = offsetof(HeaderObject, reduction_date)
    },
    {
        .name = "lambda_offset",
        .type = Py_T_FLOAT,
        .offset = offsetof(HeaderObject, lambda_offset)
    },
    {
        .name = "beta_offset",
        .type = Py_T_FLOAT,
        .offset = offsetof(HeaderObject, beta_offset)
    },
    {
        .name = "coordinate_system",
        .type = Py_T_INT,
        .offset = offsetof(HeaderObject, coordinate_system)
    },
    {
        .name = "kind",
        .type = Py_T_INT,
        .offset = offsetof(HeaderObject, kind)
    },
    {
        .name = "quality",
        .type = Py_T_INT,
        .offset = offsetof(HeaderObject, quality)
    },
    {
        .name = "position_angle",
        .type = Py_T_FLOAT,
        .offset = offsetof(HeaderObject, position_angle)
    },
    {
        .name = "scan",
        .type = Py_T_LONG,
        .offset = offsetof(HeaderObject, scan)
    },
    {
        .name = "sub_scan",
        .type = Py_T_INT,
        .offset = offsetof(HeaderObject, sub_scan)
    },
    {NULL}
};

PyGetSetDef Header_getset[] = {
    {
        .name = "source",
        .get = (getter) Header_get_source,
        .set = (setter) Header_set_source
    },
    {
        .name = "line",
        .get = (getter) Header_get_line,
        .set = (setter) Header_set_line
    },
    {
        .name = "telescope",
        .get = (getter) Header_get_telescope,
        .set = (setter) Header_set_telescope
    },
    {NULL}
};


PyTypeObject GeneralSectionType = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "Header",
    .tp_basicsize = sizeof(HeaderObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
    .tp_traverse = (traverseproc) Header_traverse,
    .tp_clear = (inquiry) Header_clear,
    .tp_new = PyType_GenericNew,
    .tp_dealloc = (destructor) Header_dealloc,
    .tp_members = GeneralSection_members,
    .tp_getset = Header_getset
};
