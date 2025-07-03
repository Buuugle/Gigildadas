#include <python3.13/Python.h>

#include  "ContainerObject.h"
#include "HeaderObject.h"


static int exec(PyObject *module) {
    if (PyType_Ready(&ContainerType) < 0) {
        return -1;
    }
    if (PyModule_AddObjectRef(module, "Container", (PyObject *) &ContainerType) < 0) {
        return -1;
    }

    if (PyType_Ready(&HeaderType) < 0) {
        return -1;
    }
    if (PyModule_AddObjectRef(module, "Header", (PyObject *) &HeaderType) < 0) {
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
    .m_name = "gildascontainer",
    .m_size = 0,
    .m_slots = slots
};

PyMODINIT_FUNC PyInit_gildascontainer() {
    return PyModuleDef_Init(&module);
}
