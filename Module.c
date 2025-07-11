#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL Module
#include <numpy/arrayobject.h>

#include "Container.h"
#include "Header.h"
#include "Sections.h"

#define REGISTER_TYPE(TYPE, NAME)                                      \
    if (PyType_Ready(&TYPE) < 0) {                                     \
        return -1;                                                     \
    }                                                                  \
    if (PyModule_AddObjectRef(module, NAME, (PyObject *) &TYPE) < 0) { \
        return -1;                                                     \
    }

#define SET_TYPE_CONSTANT(TYPE, NAME, VALUE)         \
    PyDict_SetItemString(TYPE.tp_dict, NAME, VALUE); \
    Py_DECREF(VALUE);


static int exec(PyObject *module) {
    REGISTER_TYPE(ContainerType, "Container")
    REGISTER_TYPE(HeaderType, "Header")

    REGISTER_TYPE(GeneralSectionType, "GeneralSection")
    REGISTER_TYPE(PositionSectionType, "PositionSection")
    REGISTER_TYPE(SpectroSectionType, "SpectroSection")
    REGISTER_TYPE(PlotSectionType, "PlotSection")
    REGISTER_TYPE(SwitchSectionType, "SwitchSection")
    REGISTER_TYPE(CalibrationSectionType, "CalibrationSection")

    SET_TYPE_CONSTANT(GeneralSectionType, "ID", PyLong_FromLong(GENERAL_SECTION_ID))
    SET_TYPE_CONSTANT(PositionSectionType, "ID", PyLong_FromLong(POSITION_SECTION_ID))
    SET_TYPE_CONSTANT(SpectroSectionType, "ID", PyLong_FromLong(SPECTRO_SECTION_ID))
    SET_TYPE_CONSTANT(PlotSectionType, "ID", PyLong_FromLong(PLOT_SECTION_ID))
    SET_TYPE_CONSTANT(SwitchSectionType, "ID", PyLong_FromLong(SWITCH_SECTION_ID))
    SET_TYPE_CONSTANT(CalibrationSectionType, "ID", PyLong_FromLong(CALIBRATION_SECTION_ID))
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
    .m_name = "container",
    .m_size = 0,
    .m_slots = slots
};

PyMODINIT_FUNC PyInit_container() {
    import_array()
    return PyModuleDef_Init(&module);
}
