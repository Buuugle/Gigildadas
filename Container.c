#include <stddef.h>
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL Module
#include <numpy/arrayobject.h>

#include "Container.h"
#include "Header.h"
#include "Utils.h"

#define NO_INPUT_FILE_ERROR PyErr_SetString(PyExc_AttributeError, "no input file")


int Container_traverse(ContainerObject *self,
                       visitproc visit,
                       void *arg) {
    // Py_VISIT
    return 0;
}

int Container_clear(ContainerObject *self) {
    if (self->input_file) {
        fclose(self->input_file);
    }
    self->input_file = NULL;
    if (self->output_file) {
        fclose(self->output_file);
    }
    self->output_file = NULL;

    PyMem_Free(self->extension_records);
    self->extension_records = NULL;

    return 0;
}

void Container_dealloc(ContainerObject *self) {
    PyObject_GC_UnTrack(self);
    Container_clear(self);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

PyObject *Container_set_input(ContainerObject *self,
                              PyObject *args) {
    char *filename;
    if (!PyArg_ParseTuple(args, "s", &filename)) {
        return NULL;
    }
    if (self->input_file) {
        fclose(self->input_file);
    }

    self->input_file = fopen(filename, "rb");
    if (!self->input_file) {
        PyErr_SetString(PyExc_IOError, "file could not be opened");
        return NULL;
    }

    flockfile(self->input_file);
    fseek(self->input_file, 0, SEEK_SET);
    fread_unlocked(&self->file_version,
                   offsetof(ContainerObject, extension_records) - offsetof(ContainerObject, file_version), 1,
                   self->input_file);

    void *temp = PyMem_Realloc(self->extension_records, self->extension_count * sizeof(long));
    if (!temp) {
        MEMORY_ALLOCATION_ERROR;
        funlockfile(self->input_file);
        return NULL;
    }
    self->extension_records = temp;
    fread_unlocked(self->extension_records,
                   sizeof(long), self->extension_count,
                   self->input_file);

    funlockfile(self->input_file);
    return Py_NewRef(Py_None);
}

PyObject *Container_get_headers(const ContainerObject *self,
                                PyObject *args) {
    if (!self->extension_records) {
        NO_INPUT_FILE_ERROR;
        return NULL;
    }

    long entry_count = self->next_entry - 1;
    Py_ssize_t start = 0;
    Py_ssize_t end = entry_count;
    if (!PyArg_ParseTuple(args, "|ll", &start, &end)) {
        return NULL;
    }
    if (start >= end || start < 0 || end > entry_count) {
        PyErr_SetString(PyExc_IndexError, "range out of bound");
        return NULL;
    }
    entry_count = end - start;

    PyObject *list = PyList_New(entry_count);
    if (!list) {
        MEMORY_ALLOCATION_ERROR;
        return NULL;
    }

    long cumul_size = 0;
    const long header_size = offsetof(HeaderObject, identifier) - offsetof(HeaderObject, descriptor_record);
    HeaderObject *entries[entry_count];
    size_t index = 0;
    flockfile(self->input_file);

    for (int i = 0; i < self->extension_count; ++i) {
        const long size = (long) ceil(
            self->extension_length_init * power((double) self->extension_length_power / 10., i)
        );
        const long next_cumul = cumul_size + size;
        if (start < next_cumul) {
            const long p_start = max(0, start - cumul_size);
            const long p_end = min(size, end - cumul_size);
            if (p_start < p_end) {
                fseek(self->input_file,
                      WORD_LENGTH * self->record_length * (self->extension_records[i] - 1)
                      + p_start * header_size,
                      SEEK_SET);
                for (long j = p_start; j < p_end; ++j) {
                    entries[index] = (HeaderObject *) GeneralSectionType.tp_alloc(&GeneralSectionType, 1);
                    fread_unlocked(&entries[index]->descriptor_record,
                                   header_size, 1,
                                   self->input_file);
                    ++index;
                }
            }
            if (end <= next_cumul) {
                break;
            }
        }
        cumul_size = next_cumul;
    }

    const long descriptor_size = offsetof(HeaderObject, section_identifiers) - offsetof(HeaderObject, identifier);
    for (int i = 0; i < entry_count; ++i) {
        HeaderObject *entry = entries[i];
        const long descriptor_address = self->record_length * (entry->descriptor_record - 1)
                                        + entry->descriptor_word - 1;

        fseek(self->input_file,
              descriptor_address * WORD_LENGTH,
              SEEK_SET);
        fread_unlocked(&entry->identifier,
                       descriptor_size, 1,
                       self->input_file);

        entry->section_identifiers = PyMem_Malloc(entry->section_count * sizeof(int));
        entry->section_lengths = PyMem_Malloc(entry->section_count * sizeof(long));
        entry->section_addresses = PyMem_Malloc(entry->section_count * sizeof(long));
        if (!entry->section_identifiers ||
            !entry->section_lengths ||
            !entry->section_addresses) {
            MEMORY_ALLOCATION_ERROR;
            Py_DECREF(list);
            funlockfile(self->input_file);
            return NULL;
        }
        fread_unlocked(entry->section_identifiers,
                       sizeof(int), entry->section_count,
                       self->input_file);
        fread_unlocked(entry->section_lengths,
                       sizeof(long), entry->section_count,
                       self->input_file);
        fread_unlocked(entry->section_addresses,
                       sizeof(long), entry->section_count,
                       self->input_file);
        PyList_SET_ITEM(list, i, entry);
        entries[i] = NULL;
    }

    funlockfile(self->input_file);
    return list;
}

PyObject *Container_get_entry_count(const ContainerObject *self,
                                    PyObject *Py_UNUSED(ignored)) {
    if (!self->extension_records) {
        NO_INPUT_FILE_ERROR;
        return NULL;
    }
    return PyLong_FromLong(self->next_entry - 1);
}

PyObject *Container_get_data(const ContainerObject *self,
                             PyObject *args) {
    if (!self->extension_records) {
        NO_INPUT_FILE_ERROR;
        return NULL;
    }

    PyObject *arg;
    if (!PyArg_ParseTuple(args, "O", &arg)) {
        return NULL;
    }

    PyObject *sequence = PySequence_Fast(arg, "argument is not a sequence");
    if (!sequence) {
        return NULL;
    }

    const npy_intp entry_count = PySequence_Fast_GET_SIZE(sequence);
    PyObject **entries = PySequence_Fast_ITEMS(sequence);
    npy_intp data_length = 0;
    for (npy_intp i = 0; i < entry_count; ++i) {
        PyObject *object = entries[i];
        if (!PyObject_TypeCheck(object, &GeneralSectionType)) {
            Py_DECREF(sequence);
            PyErr_SetString(PyExc_TypeError, "element is not a Header");
            return NULL;
        }
        const HeaderObject *entry = (HeaderObject *) object;
        if (entry->data_length > data_length) {
            data_length = entry->data_length;
        }
    }

    float *data = calloc(entry_count * data_length, sizeof(float));
    if (!data) {
        MEMORY_ALLOCATION_ERROR;
        return NULL;
    }
    flockfile(self->input_file);
    for (int i = 0; i < entry_count; ++i) {
        HeaderObject *entry = (HeaderObject *) entries[i];
        const long descriptor_address = self->record_length * (entry->descriptor_record - 1)
                                        + entry->descriptor_word - 1;
        fseek(self->input_file,
              (descriptor_address + entry->data_address - 1) * WORD_LENGTH,
              SEEK_SET);
        fread_unlocked(data + i * data_length,
                       sizeof(float), entry->data_length,
                       self->input_file);
    }
    funlockfile(self->input_file);
    Py_DECREF(sequence);
    sequence = NULL;

    const npy_intp dims[] = {entry_count, data_length};
    PyObject *array = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, data);
    if (!array) {
        free(data);
        MEMORY_ALLOCATION_ERROR;
        return NULL;
    }
    PyArray_ENABLEFLAGS((PyArrayObject *) array, NPY_ARRAY_OWNDATA);

    return array;
}

PyMethodDef Container_methods[] = {
    {
        .ml_name = "set_input",
        .ml_meth = (PyCFunction) Container_set_input,
        .ml_flags = METH_VARARGS
    },
    {
        .ml_name = "get_entry_count",
        .ml_meth = (PyCFunction) Container_get_entry_count,
        .ml_flags = METH_NOARGS
    },
    {
        .ml_name = "get_headers",
        .ml_meth = (PyCFunction) Container_get_headers,
        .ml_flags = METH_VARARGS
    },
    {
        .ml_name = "get_data",
        .ml_meth = (PyCFunction) Container_get_data,
        .ml_flags = METH_VARARGS
    },
    {NULL}
};

PyTypeObject ContainerType = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "gigildas.Container",
    .tp_basicsize = sizeof(ContainerObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
    .tp_new = PyType_GenericNew,
    .tp_dealloc = (destructor) Container_dealloc,
    .tp_traverse = (traverseproc) Container_traverse,
    .tp_clear = (inquiry) Container_clear,
    .tp_methods = Container_methods,
};
