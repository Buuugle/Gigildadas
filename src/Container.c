#include <stddef.h>
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL Module
#include <numpy/arrayobject.h>

#include "Container.h"
#include "Utils.h"
#include "Header.h"
#include "Sections.h"

#define NO_INPUT_FILE_ERROR PyErr_SetString(PyExc_AttributeError, "no input file")
#define NO_HEADER_ERROR PyErr_SetString(PyExc_ValueError, "element must be a Header")
#define FILE_VERSION "2A  "
#define DESCRIPTOR_IDENTIFIER "2   "


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
                              PyObject *args,
                              PyObject *kwargs) {
    char *filename;
    static char *kwlist[] = {"filename", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s", kwlist, &filename)) {
        return NULL;
    }
    if (self->input_file) {
        fclose(self->input_file);
    }

    self->input_file = fopen(filename, "rb");
    if (!self->input_file) {
        PyErr_SetFromErrnoWithFilename(PyExc_IOError, filename);
        return NULL;
    }

    fseek(self->input_file, 0, SEEK_SET);
    if (fread(&self->file_version,
              offsetof(ContainerObject, extension_records) - offsetof(ContainerObject, file_version), 1,
              self->input_file) != 1
        || memcmp(self->file_version, FILE_VERSION, WORD_SIZE) != 0) {
        FILE_READ_ERROR;
        fclose(self->input_file);
        self->input_file = NULL;
        return NULL;
    }

    void *temp = PyMem_Realloc(self->extension_records, self->extension_count * sizeof(long));
    if (!temp) {
        MEMORY_ALLOCATION_ERROR;
        fclose(self->input_file);
        self->input_file = NULL;
        return NULL;
    }
    self->extension_records = temp;

    if (fread(self->extension_records,
              sizeof(int64_t), self->extension_count,
              self->input_file) != self->extension_count) {
        FILE_READ_ERROR;
        fclose(self->input_file);
        self->input_file = NULL;
        PyMem_Free(self->extension_records);
        self->extension_records = NULL;
        return NULL;
    }

    Py_RETURN_NONE;
}

static void cleanup_headers(HeaderObject **headers,
                            const Py_ssize_t count) {
    for (Py_ssize_t i = 0; i < count; ++i) {
        if (headers[i]) {
            Py_DECREF(headers[i]);
            headers[i] = NULL;
        }
    }
}

PyObject *Container_get_headers(const ContainerObject *self,
                                PyObject *args,
                                PyObject *kwargs) {
    if (!self->input_file) {
        NO_INPUT_FILE_ERROR;
        return NULL;
    }

    Py_ssize_t entry_count = self->next_entry - 1;
    Py_ssize_t start = 0;
    Py_ssize_t end = 0;
    static char *kwlist[] = {"start", "end", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|ll", kwlist, &start, &end)) {
        return NULL;
    }
    if (end <= start) {
        end = entry_count;
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

    Py_ssize_t cumul_size = 0;
    const Py_ssize_t header_size = offsetof(HeaderObject, identifier) - offsetof(HeaderObject, descriptor_record);
    HeaderObject *headers[entry_count];
    size_t index = 0;

    for (Py_ssize_t i = 0; i < self->extension_count; ++i) {
        const Py_ssize_t size = (Py_ssize_t) ceil(
            self->extension_length_init * power((double) self->extension_length_power / 10., i)
        );
        const Py_ssize_t next_cumul = cumul_size + size;
        if (start < next_cumul) {
            const Py_ssize_t p_start = max(0, start - cumul_size);
            const Py_ssize_t p_end = min(size, end - cumul_size);
            if (p_start < p_end) {
                fseek(self->input_file,
                      WORD_SIZE * self->record_length * (self->extension_records[i] - 1) + p_start * header_size,
                      SEEK_SET);
                for (Py_ssize_t j = p_start; j < p_end; ++j) {
                    PyObject *header = HeaderType.tp_new(&HeaderType, NULL, NULL);
                    if (!header) {
                        MEMORY_ALLOCATION_ERROR;
                        Py_DECREF(list);
                        cleanup_headers(headers, i);
                        return NULL;
                    }
                    headers[index] = (HeaderObject *) header;
                    if (fread(&headers[index]->descriptor_record, header_size, 1, self->input_file) != 1) {
                        FILE_READ_ERROR;
                        Py_DECREF(list);
                        cleanup_headers(headers, i);
                        return NULL;
                    }
                    ++index;
                }
            }
            if (end <= next_cumul) {
                break;
            }
        }
        cumul_size = next_cumul;
    }

    const Py_ssize_t descriptor_size = offsetof(HeaderObject, section_identifiers) - offsetof(HeaderObject, identifier);
    for (Py_ssize_t i = 0; i < entry_count; ++i) {
        HeaderObject *header = headers[i];
        const Py_ssize_t descriptor_address = self->record_length * (header->descriptor_record - 1)
                                              + header->descriptor_word - 1;

        fseek(self->input_file,
              descriptor_address * WORD_SIZE,
              SEEK_SET);
        if (fread(&header->identifier, descriptor_size, 1, self->input_file) != 1) {
            FILE_READ_ERROR;
            Py_DECREF(list);
            cleanup_headers(headers, entry_count);
            return NULL;
        }

        if (memcmp(header->identifier, DESCRIPTOR_IDENTIFIER, WORD_SIZE) != 0) {
            PyErr_SetString(PyExc_IndexError, "identifier not found");
            Py_DECREF(list);
            cleanup_headers(headers, entry_count);
            return NULL;
        }

        header->section_identifiers = PyMem_Malloc(header->section_count * sizeof(int32_t));
        header->section_lengths = PyMem_Malloc(header->section_count * sizeof(int64_t));
        header->section_addresses = PyMem_Malloc(header->section_count * sizeof(int64_t));
        if (!header->section_identifiers ||
            !header->section_lengths ||
            !header->section_addresses) {
            MEMORY_ALLOCATION_ERROR;
            Py_DECREF(list);
            cleanup_headers(headers, entry_count);
            return NULL;
        }

        if (fread(header->section_identifiers, sizeof(int32_t), header->section_count,
                  self->input_file) != header->section_count
            || fread(header->section_lengths, sizeof(int64_t), header->section_count,
                     self->input_file) != header->section_count
            || fread(header->section_addresses, sizeof(int64_t), header->section_count,
                     self->input_file) != header->section_count) {
            FILE_READ_ERROR;
            Py_DECREF(list);
            cleanup_headers(headers, entry_count);
            return NULL;
        }

        PyList_SET_ITEM(list, i, header);
        headers[i] = NULL;
    }

    return list;
}

PyObject *Container_get_size(const ContainerObject *self,
                             PyObject *Py_UNUSED(ignored)) {
    if (!self->input_file) {
        NO_INPUT_FILE_ERROR;
        return NULL;
    }
    return PyLong_FromLong(self->next_entry - 1);
}

PyObject *Container_get_data(const ContainerObject *self,
                             PyObject *args,
                             PyObject *kwargs) {
    if (!self->input_file) {
        NO_INPUT_FILE_ERROR;
        return NULL;
    }

    PyObject *arg;
    static char *kwlist[] = {"headers", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &arg)) {
        return NULL;
    }

    PyObject *sequence = PySequence_Fast(arg, "argument must be sequence");
    if (!sequence) {
        return NULL;
    }

    const npy_intp entry_count = PySequence_Fast_GET_SIZE(sequence);
    PyObject **headers = PySequence_Fast_ITEMS(sequence);
    npy_intp data_size = 0;
    for (npy_intp i = 0; i < entry_count; ++i) {
        PyObject *object = headers[i];
        if (!PyObject_TypeCheck(object, &HeaderType)) {
            Py_DECREF(sequence);
            NO_HEADER_ERROR;
            return NULL;
        }
        const HeaderObject *header = (HeaderObject *) object;
        if (header->data_size > data_size) {
            data_size = header->data_size;
        }
    }

    float *data = PyMem_Calloc(entry_count * data_size, sizeof(float));
    if (!data) {
        MEMORY_ALLOCATION_ERROR;
        Py_DECREF(sequence);
        return NULL;
    }

    for (Py_ssize_t i = 0; i < entry_count; ++i) {
        const HeaderObject *header = (HeaderObject *) headers[i];
        const Py_ssize_t descriptor_address = self->record_length * (header->descriptor_record - 1)
                                              + header->descriptor_word - 1;
        fseek(self->input_file,
              (descriptor_address + header->data_address - 1) * WORD_SIZE,
              SEEK_SET);
        if (fread(data + i * data_size,
                  sizeof(float), header->data_size,
                  self->input_file) != header->data_size) {
            Py_DECREF(sequence);
            FILE_READ_ERROR;
            return NULL;
        }
    }
    Py_DECREF(sequence);

    const npy_intp dims[] = {entry_count, data_size};
    PyObject *array = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, data);
    if (!array) {
        PyMem_Free(data);
        MEMORY_ALLOCATION_ERROR;
        return NULL;
    }
    PyArray_ENABLEFLAGS((PyArrayObject *) array, NPY_ARRAY_OWNDATA);
    return array;
}

PyObject *Container_get_sections(const ContainerObject *self,
                                 PyObject *args,
                                 PyObject *kwargs) {
    if (!self->input_file) {
        NO_INPUT_FILE_ERROR;
        return NULL;
    }
    PyObject *arg1;
    PyObject *arg2;
    static char *kwlist[] = {"headers", "type", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &arg1, &arg2)) {
        return NULL;
    }

    PyObject *sequence = PySequence_Fast(arg1, "argument must be sequence");
    if (!sequence) {
        return NULL;
    }

    if (!PyType_Check(arg2)) {
        PyErr_SetString(PyExc_TypeError, "arg must be a type");
        return NULL;
    }
    PyTypeObject *type = (PyTypeObject *) arg2;
    PyObject *element;
    if (!type->tp_dict || !((element = PyDict_GetItemString(type->tp_dict, "ID")))) {
        PyErr_SetString(PyExc_TypeError, "type has no ID");
        return NULL;
    }
    const int32_t id = PyLong_AsInt(element);
    if (PyErr_Occurred()) {
        return NULL;
    }
    const section_read read = get_section_read(id);
    if (!read) {
        PyErr_SetString(PyExc_ValueError, "invalid id");
        return NULL;
    }

    const Py_ssize_t entry_count = PySequence_Fast_GET_SIZE(sequence);
    PyObject *list = PyList_New(entry_count);
    if (!list) {
        MEMORY_ALLOCATION_ERROR;
        Py_DECREF(sequence);
        return NULL;
    }
    PyObject **entries = PySequence_Fast_ITEMS(sequence);

    for (Py_ssize_t i = 0; i < entry_count; ++i) {
        PyObject *object = entries[i];
        if (!PyObject_TypeCheck(object, &HeaderType)) {
            Py_DECREF(sequence);
            Py_DECREF(list);
            NO_HEADER_ERROR;
            return NULL;
        }
        const HeaderObject *header = (HeaderObject *) object;
        int section_index = -1;
        for (int j = 0; j < header->section_count; ++j) {
            if (header->section_identifiers[j] == id) {
                section_index = j;
                break;
            }
        }
        if (section_index == -1) {
            PyList_SET_ITEM(list, i, Py_None);
            continue;
        }
        PyObject *section = type->tp_new(type, NULL, NULL);
        if (!section) {
            Py_DECREF(sequence);
            Py_DECREF(list);
            MEMORY_ALLOCATION_ERROR;
            return NULL;
        }
        const Py_ssize_t descriptor_address = self->record_length * (header->descriptor_record - 1)
                                              + header->descriptor_word - 1;
        fseek(self->input_file,
              (descriptor_address + header->section_addresses[section_index] - 1) * WORD_SIZE,
              SEEK_SET);
        if (read(section, self->input_file) < 0) {
            Py_DECREF(sequence);
            Py_DECREF(list);
            Py_DECREF(section);
            return NULL;
        }
        PyList_SET_ITEM(list, i, section);
    }

    Py_DECREF(sequence);
    return list;
}

PyMethodDef Container_methods[] = {
    {
        .ml_name = "set_input",
        .ml_meth = (PyCFunction) Container_set_input,
        .ml_flags = METH_VARARGS | METH_KEYWORDS
    },
    {
        .ml_name = "get_size",
        .ml_meth = (PyCFunction) Container_get_size,
        .ml_flags = METH_NOARGS
    },
    {
        .ml_name = "get_headers",
        .ml_meth = (PyCFunction) Container_get_headers,
        .ml_flags = METH_VARARGS | METH_KEYWORDS
    },
    {
        .ml_name = "get_data",
        .ml_meth = (PyCFunction) Container_get_data,
        .ml_flags = METH_VARARGS | METH_KEYWORDS
    },
    {
        .ml_name = "get_sections",
        .ml_meth = (PyCFunction) Container_get_sections,
        .ml_flags = METH_VARARGS | METH_KEYWORDS
    },
    {NULL}
};

PyTypeObject ContainerType = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "Container",
    .tp_basicsize = sizeof(ContainerObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
    .tp_new = PyType_GenericNew,
    .tp_dealloc = (destructor) Container_dealloc,
    .tp_traverse = (traverseproc) Container_traverse,
    .tp_clear = (inquiry) Container_clear,
    .tp_methods = Container_methods
};
