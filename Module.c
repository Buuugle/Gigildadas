#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL Module
#include <numpy/arrayobject.h>

#include "Container.h"
#include "Header.h"
#include "Sections.h"


static int exec(PyObject *module) {
    if (PyType_Ready(&ContainerType) < 0) {
        return -1;
    }
    if (PyModule_AddObjectRef(module, "Container", (PyObject *) &ContainerType) < 0) {
        return -1;
    }

    if (PyType_Ready(&GeneralSectionType) < 0) {
        return -1;
    }
    if (PyModule_AddObjectRef(module, "Header", (PyObject *) &GeneralSectionType) < 0) {
        return -1;
    }
    PyObject_SetAttrString(module, "GENERAL_SECTION_ID", PyLong_FromLong(GENERAL_SECTION_ID));
    PyObject_SetAttrString(module, "POSITION_SECTION_ID", PyLong_FromLong(POSITION_SECTION_ID));
    PyObject_SetAttrString(module, "SPECTRO_SECTION_ID", PyLong_FromLong(SPECTRO_SECTION_ID));
    PyObject_SetAttrString(module, "PLOT_SECTION_ID", PyLong_FromLong(PLOT_SECTION_ID));
    PyObject_SetAttrString(module, "SWITCH_SECTION_ID", PyLong_FromLong(SWITCH_SECTION_ID));
    PyObject_SetAttrString(module, "CALIBRATION_SECTION_ID", PyLong_FromLong(CALIBRATION_SECTION_ID));
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
    .m_name = "gigildas",
    .m_size = 0,
    .m_slots = slots
};

PyMODINIT_FUNC PyInit_gigildas() {
    import_array()
    return PyModuleDef_Init(&module);
}
