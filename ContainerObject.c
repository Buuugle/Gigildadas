#include <stddef.h>

#include "ContainerObject.h"


static double power(const double x,
                    const int n) {
    double result = 1.;
    for (int i = 0; i < n; ++i) {
        result *= x;
    }
    return result;
}


static void free_entry_descriptors_content(struct EntryDescriptor *entry_descriptors,
                                           const long entry_count) {
    if (!entry_descriptors) {
        return;
    }
    for (int i = 0; i < entry_count; ++i) {
        struct EntryDescriptor *descriptor = entry_descriptors + i;
        free(descriptor->section_identifiers);
        descriptor->section_identifiers = NULL;
        free(descriptor->section_lengths);
        descriptor->section_lengths = NULL;
        free(descriptor->section_addresses);
        descriptor->section_addresses = NULL;
        free(descriptor->data);
        descriptor->data = NULL;
    }
}

static PyObject *unicode_from_chars(const char *str,
                                    const long length) {
    char new_str[length + 1];
    new_str[length] = '\0';
    for (int i = 0; i < length; ++i) {
        new_str[i] = str[i];
    }
    return PyUnicode_FromString(new_str);
}

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

    struct FileHeader *file_header = &self->file_header;
    const long entry_count = file_header->next_entry - 1;

    free_entry_descriptors_content(self->entry_descriptors, entry_count);
    free(self->entry_descriptors);
    self->entry_descriptors = NULL;
    free(self->entry_headers);
    self->entry_headers = NULL;

    free(file_header->extension_records);
    file_header->extension_records = NULL;

    return 0;
}

void Container_dealloc(ContainerObject *self) {
    PyObject_GC_UnTrack(self);
    Container_clear(self);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

PyObject *Container_new(PyTypeObject *type,
                        PyObject *args,
                        PyObject *kwargs) {
    ContainerObject *self = (ContainerObject *) type->tp_alloc(type, 0);
    if (!self) {
        return NULL;
    }

    self->input_file = NULL;
    self->output_file = NULL;
    self->file_header.extension_records = NULL;
    self->entry_headers = NULL;
    self->entry_descriptors = NULL;

    return (PyObject *) self;
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
        PyErr_SetString(PyExc_IOError, "File could not be opened");
        return NULL;
    }

    return Py_NewRef(Py_None);
}

PyObject *Container_read_headers(ContainerObject *self,
                                 PyObject *Py_UNUSED(ignored)) {
    if (!self->input_file) {
        PyErr_SetString(PyExc_AttributeError, "No input file opened");
        return NULL;
    }

    struct FileHeader *file_header = &self->file_header;

    fseek(self->input_file, 0, SEEK_SET);
    fread(file_header,
          offsetof(struct FileHeader, extension_records), 1,
          self->input_file);

    void *temp = realloc(file_header->extension_records, file_header->extension_count * sizeof(long));
    if (!temp) {
        PyErr_SetString(PyExc_MemoryError, "Cannot allocate memory for extension_records");
        return NULL;
    }
    file_header->extension_records = temp;
    fread(file_header->extension_records, sizeof(long), file_header->extension_count, self->input_file);

    const long entry_count = file_header->next_entry - 1;

    temp = realloc(self->entry_headers, entry_count * sizeof(struct EntryHeader));
    if (!temp) {
        PyErr_SetString(PyExc_MemoryError, "Cannot allocate memory for entry_headers");
        return NULL;
    }
    self->entry_headers = temp;

    long current_count = 0;
    for (int i = 0; i < file_header->extension_count; ++i) {
        long count = (long) ceil(
            file_header->extension_length_init * power((double) file_header->extension_length_power / 10., i)
        );
        const long max_count = entry_count - current_count;
        if (count > max_count) {
            count = max_count;
        }
        fseek(self->input_file,
              WORD_LENGTH * file_header->record_length * (file_header->extension_records[i] - 1),
              SEEK_SET);
        fread(self->entry_headers + current_count,
              sizeof(struct EntryHeader),
              count,
              self->input_file);

        current_count += count;
    }

    return Py_NewRef(Py_None);
}

PyObject *Container_get_size(const ContainerObject *self,
                             PyObject *Py_UNUSED(ignored)) {
    if (!self->entry_headers) {
        PyErr_SetString(PyExc_AttributeError, "No headers found");
        return NULL;
    }
    return PyLong_FromLong(self->file_header.next_entry - 1);
}

PyObject *Container_get_headers(const ContainerObject *self,
                                PyObject *args) {
    if (!self->entry_headers) {
        PyErr_SetString(PyExc_AttributeError, "No headers found");
        return NULL;
    }

    long begin = 0;
    long end = self->file_header.next_entry - 1;
    if (!PyArg_ParseTuple(args, "|ll", &begin, &end)) {
        return NULL;
    }

    PyObject *list = PyList_New(end - begin);
    if (!list) {
        PyErr_SetString(PyExc_ValueError, "Cannot create list");
        return NULL;
    }

    PyObject *header_args = PyTuple_New(0);
    PyObject *header_kwargs = PyDict_New();
    const long char_length = 3 * WORD_LENGTH;
    for (Py_ssize_t i = begin; i < end; ++i) {
        HeaderObject *header = (HeaderObject *) Header_new(&HeaderType, header_args, header_kwargs);
        const struct EntryHeader *entry_header = self->entry_headers + i;
        header->number = entry_header->number;
        header->version = entry_header->version;
        Py_SETREF(header->source, unicode_from_chars(entry_header->source, char_length));
        Py_SETREF(header->line, unicode_from_chars(entry_header->line, char_length));
        Py_SETREF(header->telescope, unicode_from_chars(entry_header->telescope, char_length));
        header->observation_date = entry_header->observation_date;
        header->reduction_date = entry_header->reduction_date;
        header->lambda_offset = entry_header->lambda_offset;
        header->beta_offset = entry_header->beta_offset;
        header->coordinate_system = entry_header->coordinate_system;
        header->kind = entry_header->kind;
        header->quality = entry_header->quality;
        header->position_angle = entry_header->position_angle;
        header->scan = entry_header->scan;
        header->sub_scan = entry_header->sub_scan;

        PyList_SET_ITEM(list, i, (PyObject *) header);
    }
    Py_DECREF(header_args);
    Py_DECREF(header_kwargs);

    return list;
}

PyObject *Container_read_descriptors(ContainerObject *self,
                                     PyObject *Py_UNUSED(ignored)) {
    if (!self->entry_headers) {
        PyErr_SetString(PyExc_AttributeError, "No headers found");
        return NULL;
    }

    const struct FileHeader *file_header = &self->file_header;
    const long entry_count = file_header->next_entry - 1;

    free_entry_descriptors_content(self->entry_descriptors, entry_count);
    void *temp = realloc(self->entry_descriptors, entry_count * sizeof(struct EntryDescriptor));
    if (!temp) {
        PyErr_SetString(PyExc_MemoryError, "Cannot allocate memory for entry_descriptors");
        return NULL;
    }
    self->entry_descriptors = temp;

    for (int i = 0; i < entry_count; ++i) {
        const struct EntryHeader *entry_header = self->entry_headers + i;
        const long descriptor_address = file_header->record_length * (entry_header->descriptor_record - 1)
                                        + entry_header->descriptor_word - 1;
        struct EntryDescriptor *descriptor = self->entry_descriptors + i;

        fseek(self->input_file,
              descriptor_address * WORD_LENGTH,
              SEEK_SET);
        fread(descriptor,
              offsetof(struct EntryDescriptor, section_identifiers), 1,
              self->input_file);

        descriptor->section_identifiers = malloc(descriptor->section_count * sizeof(int));
        if (!descriptor->section_identifiers) {
            PyErr_SetString(PyExc_MemoryError, "Cannot allocate memory for section_identifiers");
            return NULL;
        }
        fread(descriptor->section_identifiers,
              sizeof(int), descriptor->section_count,
              self->input_file);

        descriptor->section_lengths = malloc(descriptor->section_count * sizeof(long));
        if (!descriptor->section_lengths) {
            PyErr_SetString(PyExc_MemoryError, "Cannot allocate memory for section_lengths");
            return NULL;
        }
        fread(descriptor->section_lengths,
              sizeof(long), descriptor->section_count,
              self->input_file);

        descriptor->section_addresses = malloc(descriptor->section_count * sizeof(long));
        if (!descriptor->section_addresses) {
            PyErr_SetString(PyExc_MemoryError, "Cannot allocate memory for section_addresses");
            return NULL;
        }
        fread(descriptor->section_addresses,
              sizeof(long), descriptor->section_count,
              self->input_file);

        descriptor->data = malloc(descriptor->data_length * sizeof(float));
        if (!descriptor->data) {
            PyErr_SetString(PyExc_MemoryError, "Cannot allocate memory for data");
            return NULL;
        }
        fseek(self->input_file,
              (descriptor_address + descriptor->data_address - 1) * WORD_LENGTH,
              SEEK_SET);
        fread(descriptor->data,
              sizeof(float),
              descriptor->data_length,
              self->input_file);
    }

    return Py_NewRef(Py_None);
}

PyMethodDef Container_methods[] = {
    {
        .ml_name = "set_input",
        .ml_meth = (PyCFunction) Container_set_input,
        .ml_flags = METH_VARARGS
    },
    {
        .ml_name = "read_headers",
        .ml_meth = (PyCFunction) Container_read_headers,
        .ml_flags = METH_NOARGS
    },
    {
        .ml_name = "get_size",
        .ml_meth = (PyCFunction) Container_get_size,
        .ml_flags = METH_NOARGS
    },
    {
        .ml_name = "get_headers",
        .ml_meth = (PyCFunction) Container_get_headers,
        .ml_flags = METH_VARARGS
    },
    {
        .ml_name = "read_descriptors",
        .ml_meth = (PyCFunction) Container_read_descriptors,
        .ml_flags = METH_NOARGS
    },
    {NULL}
};

PyTypeObject ContainerType = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "gildascontainer.Container",
    .tp_basicsize = sizeof(ContainerObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
    .tp_new = Container_new,
    .tp_dealloc = (destructor) Container_dealloc,
    .tp_traverse = (traverseproc) Container_traverse,
    .tp_clear = (inquiry) Container_clear,
    .tp_methods = Container_methods,
};
