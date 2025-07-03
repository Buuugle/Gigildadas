#define PY_SSIZE_T_CLEAN
#include <python3.13/Python.h>

#include <stddef.h>

#include  "FileEditor.h"


static int exec(PyObject *module) {
    if (PyType_Ready(&FileEditorType) < 0) {
        return -1;
    }
    if (PyModule_AddObjectRef(module, "FileEditor", (PyObject *) &FileEditorType) < 0) {
        return -1;
    }

    return 0;
}

static PyModuleDef_Slot slots[] = {
    {
        .slot = Py_mod_exec,
        .value = exec
    },
    {
        .slot = Py_mod_multiple_interpreters,
        .value = Py_MOD_MULTIPLE_INTERPRETERS_NOT_SUPPORTED
    },
    {0, NULL}
};

static PyModuleDef module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "classeditor",
    .m_size = 0,
    .m_slots = slots
};

PyMODINIT_FUNC PyInit_classeditor() {
    return PyModuleDef_Init(&module);
}
