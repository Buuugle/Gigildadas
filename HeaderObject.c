#include <stddef.h>

#include "HeaderObject.h"


int Header_traverse(const HeaderObject *self,
                    const visitproc visit,
                    void *arg) {
    Py_VISIT(self->source);
    Py_VISIT(self->line);
    Py_VISIT(self->telescope);
    return 0;
}

int Header_clear(HeaderObject *self) {
    Py_CLEAR(self->source);
    Py_CLEAR(self->line);
    Py_CLEAR(self->telescope);
    return 0;
}

void Header_dealloc(HeaderObject *self) {
    PyObject_GC_UnTrack(self);
    Header_clear(self);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

PyObject *Header_new(PyTypeObject *type,
                     PyObject *args,
                     PyObject *kwargs) {
    HeaderObject *self = (HeaderObject *) type->tp_alloc(type, 0);
    if (!self) {
        return NULL;
    }

    self->source = PyUnicode_FromString("");
    self->line = PyUnicode_FromString("");
    self->telescope = PyUnicode_FromString("");
    if (!(self->source && self->line && self->telescope)) {
        Py_DECREF(self);
        return NULL;
    }

    return (PyObject *) self;
}

int Header_set_source(HeaderObject *self,
                      PyObject *value,
                      void *closure) {
    if (!value) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete source");
        return -1;
    }
    if (!PyUnicode_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Source must be a string");
        return -1;
    }
    Py_SETREF(self->source, Py_NewRef(value));
    return 0;
}

PyObject *Header_get_source(const HeaderObject *self,
                            void *closure) {
    return Py_NewRef(self->source);
}

int Header_set_line(HeaderObject *self,
                    PyObject *value,
                    void *closure) {
    if (!value) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete line");
        return -1;
    }
    if (!PyUnicode_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Line must be a string");
        return -1;
    }
    Py_SETREF(self->line, Py_NewRef(value));
    return 0;
}

PyObject *Header_get_line(const HeaderObject *self,
                          void *closure) {
    return Py_NewRef(self->line);
}

int Header_set_telescope(HeaderObject *self,
                         PyObject *value,
                         void *closure) {
    if (!value) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete telescope");
        return -1;
    }
    if (!PyUnicode_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Telescope must be a string");
        return -1;
    }
    Py_SETREF(self->telescope, Py_NewRef(value));
    return 0;
}

PyObject *Header_get_telescope(const HeaderObject *self,
                               void *closure) {
    return Py_NewRef(self->telescope);
}

PyMemberDef Header_members[] = {
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

extern PyTypeObject HeaderType = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "Header",
    .tp_basicsize = sizeof(HeaderObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
    .tp_new = Header_new,
    .tp_dealloc = (destructor) Header_dealloc,
    .tp_traverse = (traverseproc) Header_traverse,
    .tp_clear = (inquiry) Header_clear,
    .tp_members = Header_members,
    .tp_getset = Header_getset
};
