#include "FileEditor.h"

#include <stddef.h>


static double power(const double x,
                    const int n) {
    double result = 1.;
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
    // TODO: free all pointers
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
    self->entry_descriptors = NULL;

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
    fread(file_header,
          offsetof(struct FileHeader, extension_records), 1,
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

PyObject *FileEditor_read_entries(FileEditor *self,
                                  PyObject *Py_UNUSED(ignored)) {
    const struct FileHeader *file_header = &self->file_header;
    const long entry_count = file_header->next_entry - 1;

    self->entry_headers = reallocarray(self->entry_headers,
                                       entry_count,
                                       sizeof(struct EntryHeader));
    if (!self->entry_headers) {
        PyErr_SetString(PyExc_MemoryError, "Cannot allocate memory for entry_headers");
        return NULL;
    }

    long current_count = 0;
    for (int i = 0; i < file_header->extension_count; ++i) {
        long count = (long) ceil(
            file_header->extension_length_init * power((double) file_header->extension_length_power / 10., i)
        );
        if (count > entry_count - current_count) {
            count = current_count;
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


PyObject *FileEditor_read_data(FileEditor *self,
                               PyObject *Py_UNUSED(ignored)) {
    const struct FileHeader *file_header = &self->file_header;
    const long entry_count = file_header->next_entry - 1;

    self->entry_descriptors = reallocarray(self->entry_descriptors,
                                           entry_count,
                                           sizeof(struct EntryDescriptor));
    if (!self->entry_descriptors) {
        PyErr_SetString(PyExc_MemoryError, "Cannot allocate memory for entry_descriptors");
        return NULL;
    }

    for (int i = 0; i < entry_count; ++i) {
        const struct EntryHeader *entry_header = self->entry_headers + i;
        const long descriptor_address = file_header->record_length * (entry_header->descriptor_record - 1)
                                        + entry_header->descriptor_word - 1;
        struct EntryDescriptor *entry_descriptor = self->entry_descriptors + i;

        fseek(self->input_file,
              descriptor_address * WORD_LENGTH,
              SEEK_SET);
        fread(entry_descriptor,
              offsetof(struct EntryDescriptor, section_identifiers), 1,
              self->input_file);

        fread(entry_descriptor->section_identifiers,
              sizeof(int), entry_descriptor->section_count,
              self->input_file);
        fread(entry_descriptor->section_lengths,
              sizeof(long), entry_descriptor->section_count,
              self->input_file);
        fread(entry_descriptor->section_addresses,
              sizeof(long), entry_descriptor->section_count,
              self->input_file);

        fseek(self->input_file,
              (descriptor_address + entry_descriptor->data_address - 1) * WORD_LENGTH,
              SEEK_SET);
        fread(entry_descriptor->data,
              sizeof(float),
              entry_descriptor->data_length,
              self->input_file);
    }

    return Py_NewRef(Py_None);
}
