#define PY_SSIZE_T_CLEAN
#include <python3.13/Python.h>

#include <stddef.h>


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
        .ml_doc = PyDoc_STR("Je test")
    },
    {NULL}
};

typedef struct {
    PyObject_HEAD

    PyObject *name;
    int number;
} TestObject;

static void Test_dealloc(TestObject *self) {
    Py_XDECREF(self->name);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject *Test_new(PyTypeObject *type,
                          PyObject *args,
                          PyObject *kwargs) {
    TestObject *self = (TestObject *) type->tp_alloc(type, 0);
    if (self != NULL) {
        self->name = PyUnicode_FromString("");
        if (self->name == NULL) {
            Py_DECREF(self);
            return NULL;
        }
        self->number = 0;
    }
    return (PyObject *) self;
}

static int Test_init(TestObject *self,
                     PyObject *args,
                     PyObject *kwargs) {
    static char *kwlist[] = {"name", "number", NULL};
    PyObject *name = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|Oi", kwlist, &name, &self->number)) {
        return -1;
    }
    if (name) {
        Py_XSETREF(self->name, Py_NewRef(name));
    }
    return 0;
}

static PyMemberDef Test_members[] = {
    {
        .name = "number",
        .type = Py_T_INT,
        .offset = offsetof(TestObject, number),
        .flags = 0,
        .doc = PyDoc_STR("Number test")
    },
    {NULL}
};

static PyObject *Test_get_name(const TestObject *self,
                               void *closure) {
    return Py_NewRef(self->name);
}

static int Test_set_name(TestObject *self,
                         PyObject *value,
                         void *closure) {
    if (value == NULL) {
        PyErr_SetString(PyExc_ValueError, "Cannot delete name");
        return -1;
    }
    if (!PyUnicode_Check(value)) {
        PyErr_SetString(PyExc_ValueError, "Test value is not a string");
        return -1;
    }
    Py_SETREF(self->name, Py_NewRef(value));
    return 0;
}

static PyGetSetDef Test_getsets[] = {
    {
        .name = "name",
        .get = (getter) Test_get_name,
        .set = (setter) Test_set_name,
        .doc = PyDoc_STR("Set name"),
        .closure = NULL
    },
    {NULL}
};

static PyObject *Test_display(const TestObject *self,
                              PyObject *args) {
    if (self->name == NULL) {
        PyErr_SetString(PyExc_AttributeError, "name");
        return NULL;
    }
    int x;
    int y;
    if (!PyArg_ParseTuple(args, "ii", &x, &y)) {
        return NULL;
    }
    printf("Test number: %d\n", self->number);
    return PyUnicode_FromFormat("%S %d", self->name, x * y);
}

static PyMethodDef Test_methods[] = {
    {
        .ml_name = "display",
        .ml_meth = (PyCFunction) Test_display,
        .ml_flags = METH_VARARGS,
        .ml_doc = PyDoc_STR("Test display")
    },
    {NULL}
};

static PyTypeObject TestType = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "test.Test",
    .tp_doc = PyDoc_STR("Je test"),
    .tp_basicsize = sizeof(TestObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = Test_new,
    .tp_init = (initproc) Test_init,
    .tp_dealloc = (destructor) Test_dealloc,
    .tp_members = Test_members,
    .tp_methods = Test_methods,
    .tp_getset = Test_getsets
};

static int exec(PyObject *module) {
    if (PyType_Ready(&TestType) < 0) {
        return -1;
    }

    if (PyModule_AddObjectRef(module, "Test", (PyObject *) &TestType) < 0) {
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
    .m_name = "test",
    .m_doc = PyDoc_STR("Testouille"),
    .m_size = 0,
    .m_methods = methods,
    .m_slots = slots
};

PyMODINIT_FUNC PyInit_test() {
    return PyModuleDef_Init(&module);
}
