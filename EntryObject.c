#include <stddef.h>

#include "EntryObject.h"


int Entry_traverse(EntryObject *self,
                   visitproc visit,
                   void *arg) {
    return 0;
}

int Entry_clear(EntryObject *self) {
    PyMem_Free(self->section_identifiers);
    PyMem_Free(self->section_lengths);
    PyMem_Free(self->section_addresses);
    PyMem_Free(self->data);
    self->section_identifiers = NULL;
    self->section_lengths = NULL;
    self->section_addresses = NULL;
    self->data = NULL;
    return 0;
}

void Entry_dealloc(EntryObject *self) {
    PyObject_GC_UnTrack(self);
    Entry_clear(self);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

int Entry_set_source(EntryObject *self,
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
    const long length = sizeof(self->source);
    Py_ssize_t arg_length;
    wchar_t *str = PyUnicode_AsWideCharString(value, &arg_length);
    if (!str) {
        return -1;
    }
    printf("%ld\n", arg_length);
    if (arg_length < length) {
        PyErr_SetString(PyExc_ValueError, "Wrong length");
        return -1;
    }
    wcstombs(self->source, str, length);
    PyMem_Free(str);
    str = NULL;
    return 0;
}

PyObject *Entry_get_source(const EntryObject *self,
                           void *closure) {
    return unicode_from_chars(self->source, sizeof(self->source));
}

int Entry_set_line(EntryObject *self,
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
    const long length = sizeof(self->line);
    Py_ssize_t arg_length;
    wchar_t *str = PyUnicode_AsWideCharString(value, &arg_length);
    if (!str) {
        return -1;
    }
    if (arg_length < length) {
        PyErr_SetString(PyExc_ValueError, "Wrong length");
        return -1;
    }
    wcstombs(self->line, str, length);
    PyMem_Free(str);
    str = NULL;
    return 0;
}

PyObject *Entry_get_line(const EntryObject *self,
                         void *closure) {
    return unicode_from_chars(self->line, sizeof(self->line));
}

int Entry_set_telescope(EntryObject *self,
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
    const long length = sizeof(self->telescope);
    Py_ssize_t arg_length;
    wchar_t *str = PyUnicode_AsWideCharString(value, &arg_length);
    if (!str) {
        return -1;
    }
    if (arg_length < length) {
        PyErr_SetString(PyExc_ValueError, "Wrong length");
        return -1;
    }
    wcstombs(self->telescope, str, length);
    PyMem_Free(str);
    str = NULL;
    return 0;
}

PyObject *Entry_get_telescope(const EntryObject *self,
                              void *closure) {
    return unicode_from_chars(self->telescope, sizeof(self->telescope));
}

PyMemberDef Entry_members[] = {
    {
        .name = "number",
        .type = Py_T_LONG,
        .offset = offsetof(EntryObject, number)
    },
    {
        .name = "version",
        .type = Py_T_INT,
        .offset = offsetof(EntryObject, version)
    },
    {
        .name = "observation_date",
        .type = Py_T_INT,
        .offset = offsetof(EntryObject, observation_date)
    },
    {
        .name = "reduction_date",
        .type = Py_T_INT,
        .offset = offsetof(EntryObject, reduction_date)
    },
    {
        .name = "lambda_offset",
        .type = Py_T_FLOAT,
        .offset = offsetof(EntryObject, lambda_offset)
    },
    {
        .name = "beta_offset",
        .type = Py_T_FLOAT,
        .offset = offsetof(EntryObject, beta_offset)
    },
    {
        .name = "coordinate_system",
        .type = Py_T_INT,
        .offset = offsetof(EntryObject, coordinate_system)
    },
    {
        .name = "kind",
        .type = Py_T_INT,
        .offset = offsetof(EntryObject, kind)
    },
    {
        .name = "quality",
        .type = Py_T_INT,
        .offset = offsetof(EntryObject, quality)
    },
    {
        .name = "position_angle",
        .type = Py_T_FLOAT,
        .offset = offsetof(EntryObject, position_angle)
    },
    {
        .name = "scan",
        .type = Py_T_LONG,
        .offset = offsetof(EntryObject, scan)
    },
    {
        .name = "sub_scan",
        .type = Py_T_INT,
        .offset = offsetof(EntryObject, sub_scan)
    },
    {NULL}
};

PyGetSetDef Entry_getset[] = {
    {
        .name = "source",
        .get = (getter) Entry_get_source,
        .set = (setter) Entry_set_source
    },
    {
        .name = "line",
        .get = (getter) Entry_get_line,
        .set = (setter) Entry_set_line
    },
    {
        .name = "telescope",
        .get = (getter) Entry_get_telescope,
        .set = (setter) Entry_set_telescope
    },
    {NULL}
};

Py_ssize_t Entry_length(const EntryObject *self) {
    if (!self->data) {
        return 0;
    }
    return self->data_length;
}

PyObject *Entry_item(const EntryObject *self,
                     const Py_ssize_t index) {
    if (!self->data) {
        PyErr_SetString(PyExc_AttributeError, "No data in entry");
        return NULL;
    }
    if (index < 0 || index >= self->data_length) {
        PyErr_SetString(PyExc_IndexError, "Index out of range");
        return NULL;
    }
    return PyFloat_FromDouble(self->data[index]);
}

int Entry_ass_item(const EntryObject *self,
                   const Py_ssize_t index,
                   PyObject *value) {
    if (!self->data) {
        PyErr_SetString(PyExc_AttributeError, "No data in entry");
        return -1;
    }
    if (index < 0 || index >= self->data_length) {
        PyErr_SetString(PyExc_IndexError, "Index out of range");
        return -1;
    }
    if (!value) {
        PyErr_SetString(PyExc_ValueError, "Cannot delete data");
        return -1;
    }
    const double temp = PyFloat_AsDouble(value);
    if (PyErr_Occurred()) {
        return -1;
    }
    self->data[index] = (float) temp;
    return 0;
}

PySequenceMethods Entry_sequence_methods = {
    .sq_length = (lenfunc) Entry_length,
    .sq_item = (ssizeargfunc) Entry_item,
    .sq_ass_item = (ssizeobjargproc) Entry_ass_item
};

PyTypeObject EntryType = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "Entry",
    .tp_basicsize = sizeof(EntryObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
    .tp_traverse = (traverseproc) Entry_traverse,
    .tp_clear = (inquiry) Entry_clear,
    .tp_new = PyType_GenericNew,
    .tp_dealloc = (destructor) Entry_dealloc,
    .tp_members = Entry_members,
    .tp_getset = Entry_getset,
    .tp_as_sequence = &Entry_sequence_methods
};
