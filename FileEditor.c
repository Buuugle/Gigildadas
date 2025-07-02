#include "FileEditor.h"

#include <stddef.h>


static double power(const double x,
                    const int n) {
    double result = x;
    for (int i = 0; i < n; ++i) {
        result *= x;
    }
    return result;
}

void FileEditor_dealloc(FileEditor *self) {
    if (self->input_file) {
        fclose(self->input_file);
    }
    if (self->output_file) {
        fclose(self->output_file);
    }
    free(self->file_header.extension_records);
    self->file_header.extension_records = NULL;
    free(self->entry_headers);
    self->entry_headers = NULL;
    Py_TYPE(self)->tp_free((PyObject *) self);
}

PyObject *FileEditor_new(PyTypeObject *type,
                         PyObject *args,
                         PyObject *kwargs) {
    FileEditor *self = (FileEditor *) type->tp_alloc(type, 0);
    if (!self) {
        return NULL;
    }

    self->input_file = NULL;
    self->output_file = NULL;
    self->file_header.extension_records = NULL;
    self->entry_headers = NULL;
    self->data = NULL;

    return (PyObject *) self;
}

int FileEditor_init(FileEditor *self,
                    PyObject *args,
                    PyObject *kwargs) {
    static char *kwlist[] = {"input_file", NULL};
    char *filename = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|s", kwlist, &filename)) {
        return -1;
    }
    if (!filename) {
        PyErr_SetString(PyExc_ValueError, "File name must be specified");
        return -1;
    }

    self->input_file = fopen(filename, "rb");
    struct FileHeader *file_header = &self->file_header;
    fread(&file_header->file_version,
          offsetof(struct FileHeader, extension_records) - offsetof(struct FileHeader, file_version),
          1,
          self->input_file);

    file_header->extension_records = reallocarray(file_header->extension_records,
                                                  file_header->extension_count,
                                                  sizeof(long));
    if (!file_header->extension_records) {
        PyErr_SetString(PyExc_MemoryError, "Cannot allocate memory for extension_records");
        return -1;
    }
    fread(file_header->extension_records, sizeof(long), file_header->extension_count, self->input_file);

    return 0;
}

PyObject *FileEditor_read_entry_headers(FileEditor *self,
                                        PyObject *Py_UNUSED(ignored)) {
    const struct FileHeader *file_header = &self->file_header;
    long entry_count = file_header->next_entry - 1;
    self->entry_headers = reallocarray(self->entry_headers, entry_count, sizeof(struct EntryHeader));
    if (!self->entry_headers) {
        PyErr_SetString(PyExc_MemoryError, "Cannot allocate memory for entry_headers");
        return NULL;
    }

    long index = 0;
    for (int i = 0; i < file_header->extension_count; ++i) {
        long length = (long) ceil(
            file_header->extension_length_init * power((double) file_header->extension_length_power / 10., i));
        if (entry_count > length) {
            entry_count -= length;
        } else {
            length = entry_count;
        }
        fread(self->entry_headers + index,
              sizeof(struct EntryHeader),
              length,
              self->input_file);
        index += length;
    }

    return Py_NewRef(Py_None);
}
