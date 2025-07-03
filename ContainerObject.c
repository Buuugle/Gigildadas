#include <stddef.h>

#include "ContainerObject.h"

#include "HeaderObject.h"
#include "Utils.h"


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

    free(file_header->extension_records);
    file_header->extension_records = NULL;

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
        PyErr_SetString(PyExc_IOError, "File could not be opened");
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

    return Py_NewRef(Py_None);
}

PyObject *Container_get_headers(const ContainerObject *self,
                                PyObject *args) {
    const struct FileHeader *file_header = &self->file_header;
    if (!file_header->extension_records) {
        PyErr_SetString(PyExc_AttributeError, "No input file");
        return NULL;
    }

    const long entry_count = file_header->next_entry - 1;
    Py_ssize_t start = 0;
    Py_ssize_t end = entry_count;
    if (!PyArg_ParseTuple(args, "|ll", &start, &end)) {
        return NULL;
    }
    if (start >= end || start < 0 || end > entry_count) {
        PyErr_SetString(PyExc_IndexError, "Invalid range");
        return NULL;
    }

    PyObject *list = PyList_New(end - start);
    if (!list) {
        PyErr_SetString(PyExc_ValueError, "Cannot create list");
        return NULL;
    }

    long cumul_size = 0;
    Py_ssize_t index = 0;
    const long entry_length = sizeof(HeaderObject) - offsetof(HeaderObject, descriptor_record);
    for (int i = 0; i < file_header->extension_count; ++i) {
        const long size = (long) ceil(
            file_header->extension_length_init * power((double) file_header->extension_length_power / 10., i)
        );
        const long next_cumul = cumul_size + size;
        if (start < next_cumul) {
            const long p_start = max(0, start - cumul_size);
            const long p_end = min(size, end - cumul_size);
            if (p_start < p_end) {
                fseek(self->input_file,
                      WORD_LENGTH * file_header->record_length * (file_header->extension_records[i] - 1)
                      + p_start * entry_length,
                      SEEK_SET);
            }
            for (long j = p_start; j < p_end; ++j) {
                HeaderObject *header = (HeaderObject *) HeaderType.tp_new(&HeaderType, NULL, NULL);
                fread(&header->descriptor_record,
                      entry_length, 1,
                      self->input_file);
                PyList_SET_ITEM(list, index, (PyObject *) header);
                ++index;
            }
            if (end <= next_cumul) {
                break;
            }
        }
        cumul_size = next_cumul;
    }

    return list;
}

PyObject *Container_get_size(const ContainerObject *self,
                             PyObject *Py_UNUSED(ignored)) {
    const struct FileHeader *file_header = &self->file_header;
    if (!file_header->extension_records) {
        PyErr_SetString(PyExc_AttributeError, "No input file");
        return NULL;
    }
    return PyLong_FromLong(file_header->next_entry - 1);
}

PyObject *Container_read_descriptors(ContainerObject *self,
                                     PyObject *Py_UNUSED(ignored)) {
    if (1) {
        PyErr_SetString(PyExc_AttributeError, "Not implemented");
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

    /*
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
    */

    return Py_NewRef(Py_None);
}

PyMethodDef Container_methods[] = {
    {
        .ml_name = "set_input",
        .ml_meth = (PyCFunction) Container_set_input,
        .ml_flags = METH_VARARGS
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
    .tp_new = PyType_GenericNew,
    .tp_dealloc = (destructor) Container_dealloc,
    .tp_traverse = (traverseproc) Container_traverse,
    .tp_clear = (inquiry) Container_clear,
    .tp_methods = Container_methods,
};
