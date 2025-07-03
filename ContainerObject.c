#include <stddef.h>

#include "ContainerObject.h"

#include "EntryObject.h"
#include "Utils.h"


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

PyObject *Container_get_entries(const ContainerObject *self,
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
    const long header_size = offsetof(EntryObject, identifier) - offsetof(EntryObject, descriptor_record);
    EntryObject *entries[entry_count];
    size_t index = 0;
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
                      + p_start * header_size,
                      SEEK_SET);
                for (long j = p_start; j < p_end; ++j) {
                    entries[index] = (EntryObject *) EntryType.tp_alloc(&EntryType, 1);
                    fread(&entries[index]->descriptor_record,
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

    const long descriptor_size = offsetof(EntryObject, section_identifiers) - offsetof(EntryObject, identifier);
    for (int i = 0; i < entry_count; ++i) {
        EntryObject *entry = entries[i];
        const long descriptor_address = file_header->record_length * (entry->descriptor_record - 1)
                                        + entry->descriptor_word - 1;

        fseek(self->input_file,
              descriptor_address * WORD_LENGTH,
              SEEK_SET);
        fread(&entry->identifier,
              descriptor_size, 1,
              self->input_file);

        entry->section_identifiers = PyMem_Malloc(entry->section_count * sizeof(int)); // TODO: PyMem_Free
        if (!entry->section_identifiers) {
            PyErr_SetString(PyExc_MemoryError, "Cannot allocate memory for section_identifiers");
            return NULL;
        }
        fread(entry->section_identifiers,
              sizeof(int), entry->section_count,
              self->input_file);

        entry->section_lengths = PyMem_Malloc(entry->section_count * sizeof(long));
        if (!entry->section_lengths) {
            PyErr_SetString(PyExc_MemoryError, "Cannot allocate memory for section_lengths");
            return NULL;
        }
        fread(entry->section_lengths,
              sizeof(long), entry->section_count,
              self->input_file);

        entry->section_addresses = PyMem_Malloc(entry->section_count * sizeof(long));
        if (!entry->section_addresses) {
            PyErr_SetString(PyExc_MemoryError, "Cannot allocate memory for section_addresses");
            return NULL;
        }
        fread(entry->section_addresses,
              sizeof(long), entry->section_count,
              self->input_file);

        PyList_SET_ITEM(list, i, entry);
    }

    return list;
}

PyObject *Container_get_entry_count(const ContainerObject *self,
                                    PyObject *Py_UNUSED(ignored)) {
    const struct FileHeader *file_header = &self->file_header;
    if (!file_header->extension_records) {
        PyErr_SetString(PyExc_AttributeError, "No input file");
        return NULL;
    }
    return PyLong_FromLong(file_header->next_entry - 1);
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
        .ml_name = "get_entries",
        .ml_meth = (PyCFunction) Container_get_entries,
        .ml_flags = METH_VARARGS
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
