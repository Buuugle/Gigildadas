#pragma once

#define PY_SSIZE_T_CLEAN
#include <python3.13/Python.h>

#define WORD_LENGTH 4


#pragma pack(1)

struct EntryDescriptor {
    char identifier[WORD_LENGTH];
    int version;
    int section_count;
    long entry_length;
    long data_address;
    long data_length;
    long number;
    int *section_identifiers;
    long *section_lengths;
    long *section_addresses;
    float *data;
};

struct EntryHeader {
    long descriptor_record;
    int descriptor_word;
    long number;
    int version;
    char source[3 * WORD_LENGTH];
    char line[3 * WORD_LENGTH];
    char telescope[3 * WORD_LENGTH];
    int observation_date;
    int reduction_date;
    float lambda_offset;
    float beta_offset;
    int coordinate_system; // code
    int kind; // code
    int quality; // code
    float position_angle;
    long scan;
    int sub_scan;
};

struct FileHeader {
    char file_version[WORD_LENGTH];
    int record_length;
    int file_kind;
    int entry_header_version;
    int entry_header_length;
    int flags;
    long next_entry;
    long next_record;
    int next_word;
    int extension_length_init;
    int extension_count;
    int extension_length_power;
    long *extension_records;
};

#pragma pack()


typedef struct FileEditor {
    PyObject_HEAD

    FILE *input_file;
    FILE *output_file;

    struct FileHeader file_header;
    struct EntryHeader *entry_headers;
    struct EntryDescriptor *entry_descriptors;
    // TODO: Sections
} FileEditor;

void FileEditor_dealloc(FileEditor *self);

PyObject *FileEditor_new(PyTypeObject *type,
                         PyObject *args,
                         PyObject *kwargs);

int FileEditor_init(FileEditor *self,
                    PyObject *args,
                    PyObject *kwargs);

PyObject *FileEditor_read_entries(FileEditor *self,
                                  PyObject *Py_UNUSED(ignored));

PyObject *FileEditor_read_data(FileEditor *self,
                               PyObject *Py_UNUSED(ignored));
