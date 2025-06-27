#include <python3.13/Python.h>

static PyObject *test(PyObject *self,
                      PyObject *args) {
    int x;
    int y;
    if (!PyArg_ParseTuple(args, "ii", &x, &y)) {
        return NULL;
    }
    return PyLong_FromLong(x + y);
}

static PyMethodDef methods[] = {
    {
        .ml_name = "test",
        .ml_meth = test,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Je test"
    },
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static PyModuleDef module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "test",
    .m_size = 0,
    .m_methods = methods,
};

PyMODINIT_FUNC PyInit_test(void) {
    return PyModuleDef_Init(&module);
}
