#include <stddef.h>

#include "Sections.h"


struct SectionReader SectionReader_get(const int id) {
    PyTypeObject *type;
    read_func read;
    switch (id) {
        case GENERAL_SECTION_ID:
            type = &GeneralSectionType;
            read = (read_func) GeneralSection_read;
            break;
        case POSITION_SECTION_ID:
            type = &PositionSectionType;
            read = (read_func) PositionSection_read;
            break;
        case SPECTRO_SECTION_ID:
            type = &SpectroSectionType;
            read = (read_func) SpectroSection_read;
            break;
        case PLOT_SECTION_ID:
            type = &PlotSectionType;
            read = (read_func) PlotSection_read;
            break;
        case SWITCH_SECTION_ID:
            type = &SwitchSectionType;
            read = (read_func) SwitchSection_read;
            break;
        case CALIBRATION_SECTION_ID:
            type = &CalibrationSectionType;
            read = (read_func) CalibrationSection_read;
            break;
        default:
            type = NULL;
            read = NULL;
    }

    const struct SectionReader reader = {type, read};
    return reader;
}

int GeneralSection_traverse(GeneralSectionObject *self,
                            visitproc visit,
                            void *arg) {
    // Py_VISIT
    return 0;
}

int GeneralSection_clear(GeneralSectionObject *self) {
    // Py_CLEAR
    return 0;
}

void GeneralSection_dealloc(GeneralSectionObject *self) {
    PyObject_GC_UnTrack(self);
    GeneralSection_clear(self);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

int GeneralSection_read(GeneralSectionObject *self,
                        FILE *file) {
    fread_unlocked(&self->ut,
                   sizeof(GeneralSectionObject) - offsetof(GeneralSectionObject, ut), 1,
                   file);
    return 0;
}


int PositionSection_traverse(PositionSectionObject *self,
                             visitproc visit,
                             void *arg) {
    // Py_VISIT
    return 0;
}

int PositionSection_clear(PositionSectionObject *self) {
    // Py_CLEAR
    return 0;
}

void PositionSection_dealloc(PositionSectionObject *self) {
    PyObject_GC_UnTrack(self);
    PositionSection_clear(self);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

PyObject *PositionSection_new(PyTypeObject *type,
                              PyObject *args,
                              PyObject *kwds) {
    PositionSectionObject *self = (PositionSectionObject *) type->tp_alloc(type, 0);
    if (self == NULL) {
        return NULL;
    }
    INIT_STRING(self->source)
    return (PyObject *) self;
}

int PositionSection_read(PositionSectionObject *self,
                         FILE *file) {
    fread_unlocked(&self->source,
                   sizeof(PositionSectionObject) - offsetof(PositionSectionObject, source), 1,
                   file);
    return 0;
}

PyObject *PositionSection_get_source(const PositionSectionObject *self,
                                     void *closure) {
    STRING_TO_UNICODE(self->source);
}

int PositionSection_set_source(PositionSectionObject *self,
                               PyObject *value,
                               void *closure) {
    UNICODE_TO_STRING(self->source, value)
}


int SpectroSection_traverse(SpectroSectionObject *self,
                            visitproc visit,
                            void *arg) {
    return 0;
}

int SpectroSection_clear(SpectroSectionObject *self) {
    return 0;
}

void SpectroSection_dealloc(SpectroSectionObject *self) {
    PyObject_GC_UnTrack(self);
    SpectroSection_clear(self);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

PyObject *SpectroSection_new(PyTypeObject *type,
                             PyObject *args,
                             PyObject *kwds) {
    SpectroSectionObject *self = (SpectroSectionObject *) type->tp_alloc(type, 0);
    if (self == NULL) {
        return NULL;
    }
    INIT_STRING(self->line)
    return (PyObject *) self;
}

int SpectroSection_read(SpectroSectionObject *self,
                        FILE *file) {
    fread_unlocked(&self->line,
                   sizeof(SpectroSectionObject) - offsetof(SpectroSectionObject, line), 1,
                   file);
    return 0;
}

PyObject *SpectroSection_get_line(const SpectroSectionObject *self,
                                  void *closure) {
    STRING_TO_UNICODE(self->line);
}

int SpectroSection_set_line(SpectroSectionObject *self,
                            PyObject *value,
                            void *closure) {
    UNICODE_TO_STRING(self->line, value);
}


int PlotSection_traverse(PlotSectionObject *self,
                         visitproc visit,
                         void *arg) {
    return 0;
}

int PlotSection_clear(PlotSectionObject *self) {
    return 0;
}

void PlotSection_dealloc(PlotSectionObject *self) {
    PyObject_GC_UnTrack(self);
    PlotSection_clear(self);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

int PlotSection_read(PlotSectionObject *self,
                     FILE *file) {
    fread_unlocked(&self->intensity_min,
                   sizeof(PlotSectionObject) - offsetof(PlotSectionObject, intensity_min), 1,
                   file);
    return 0;
}


int SwitchSection_traverse(SwitchSectionObject *self,
                           visitproc visit,
                           void *arg) {
    return 0;
}

int SwitchSection_clear(SwitchSectionObject *self) {
    PyMem_Free(self->frequency_offsets);
    PyMem_Free(self->times);
    PyMem_Free(self->weights);
    PyMem_Free(self->lambda_offsets);
    PyMem_Free(self->beta_offsets);
    self->frequency_offsets = NULL;
    self->times = NULL;
    self->weights = NULL;
    self->lambda_offsets = NULL;
    self->beta_offsets = NULL;
    return 0;
}

void SwitchSection_dealloc(SwitchSectionObject *self) {
    PyObject_GC_UnTrack(self);
    SwitchSection_clear(self);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

int SwitchSection_read(SwitchSectionObject *self,
                       FILE *file) {
    fread_unlocked(&self->phase_count,
                   sizeof(self->phase_count), 1,
                   file);
    if (SwitchSection_alloc_arrays(self) < 0) {
        return -1;
    }
    fread_unlocked(self->frequency_offsets,
                   self->phase_count * sizeof(double), 1,
                   file);
    fread_unlocked(self->times,
                   self->phase_count * sizeof(float), 1,
                   file);
    fread_unlocked(self->weights,
                   self->phase_count * sizeof(float), 1,
                   file);
    fread_unlocked(&self->switching_mode,
                   sizeof(self->switching_mode), 1,
                   file);
    fread_unlocked(self->lambda_offsets,
                   self->phase_count * sizeof(float), 1,
                   file);
    fread_unlocked(self->beta_offsets,
                   self->phase_count * sizeof(float), 1,
                   file);
    return 0;
}

int SwitchSection_alloc_arrays(SwitchSectionObject *self) {
    if (self->phase_count <= 0) {
        SwitchSection_clear(self);
        return 0;
    }

    void *temp = PyMem_Realloc(self->frequency_offsets, self->phase_count * sizeof(double));
    if (!temp) {
        MEMORY_ALLOCATION_ERROR;
        return -1;
    }
    self->frequency_offsets = temp;

    temp = PyMem_Realloc(self->times, self->phase_count * sizeof(float));
    if (!temp) {
        MEMORY_ALLOCATION_ERROR;
        return -1;
    }
    self->times = temp;

    temp = PyMem_Realloc(self->weights, self->phase_count * sizeof(float));
    if (!temp) {
        MEMORY_ALLOCATION_ERROR;
        return -1;
    }
    self->weights = temp;

    temp = PyMem_Realloc(self->lambda_offsets, self->phase_count * sizeof(float));
    if (!temp) {
        MEMORY_ALLOCATION_ERROR;
        return -1;
    }
    self->lambda_offsets = temp;

    temp = PyMem_Realloc(self->beta_offsets, self->phase_count * sizeof(float));
    if (!temp) {
        MEMORY_ALLOCATION_ERROR;
        return -1;
    }
    self->beta_offsets = temp;

    for (int i = 0; i < self->phase_count; ++i) {
        self->frequency_offsets[i] = 0.;
        self->times[i] = 0.f;
        self->weights[i] = 0.f;
        self->lambda_offsets[i] = 0.f;
        self->beta_offsets[i] = 0.f;
    }

    return 0;
}

PyObject *SwitchSection_get_phase_count(const SwitchSectionObject *self,
                                        void *closure) {
    return PyLong_FromLong(self->phase_count);
}

int SwitchSection_set_phase_count(SwitchSectionObject *self,
                                  PyObject *value,
                                  void *closure) {
    if (!value) {
        DELETE_ERROR;
        return -1;
    }
    const int temp = PyLong_AsInt(value);
    if (PyErr_Occurred()) {
        return -1;
    }
    if (temp < 0) {
        PyErr_SetString(PyExc_ValueError, "value must be positive");
        return -1;
    }
    self->phase_count = temp;
    SwitchSection_alloc_arrays(self);
    return 0;
}

PyObject *SwitchSection_get_frequency_offsets(const SwitchSectionObject *self,
                                              void *closure) {
    ARRAY_TO_TUPLE(self->frequency_offsets, self->phase_count, PyFloat_FromDouble)
}

int SwitchSection_set_frequency_offsets(SwitchSectionObject *self,
                                        PyObject *value,
                                        void *closure) {
    SEQUENCE_TO_ARRAY(self->frequency_offsets, self->phase_count, PyFloat_AsDouble, value)
}

PyObject *SwitchSection_get_times(const SwitchSectionObject *self,
                                  void *closure) {
    ARRAY_TO_TUPLE(self->times, self->phase_count, PyFloat_FromDouble)
}

int SwitchSection_set_times(SwitchSectionObject *self,
                            PyObject *value,
                            void *closure) {
    SEQUENCE_TO_ARRAY(self->times, self->phase_count, (float) PyFloat_AsDouble, value)
}

PyObject *SwitchSection_get_weights(const SwitchSectionObject *self,
                                    void *closure) {
    ARRAY_TO_TUPLE(self->weights, self->phase_count, PyFloat_FromDouble)
}

int SwitchSection_set_weights(SwitchSectionObject *self,
                              PyObject *value,
                              void *closure) {
    SEQUENCE_TO_ARRAY(self->weights, self->phase_count, (float) PyFloat_AsDouble, value)
}

PyObject *SwitchSection_get_beta_offsets(const SwitchSectionObject *self,
                                         void *closure) {
    ARRAY_TO_TUPLE(self->beta_offsets, self->phase_count, PyFloat_FromDouble)
}

int SwitchSection_set_beta_offsets(SwitchSectionObject *self,
                                   PyObject *value,
                                   void *closure) {
    SEQUENCE_TO_ARRAY(self->beta_offsets, self->phase_count, (float) PyFloat_AsDouble, value)
}

PyObject *SwitchSection_get_lambda_offsets(const SwitchSectionObject *self,
                                           void *closure) {
    ARRAY_TO_TUPLE(self->lambda_offsets, self->phase_count, PyFloat_FromDouble)
}

int SwitchSection_set_lambda_offsets(SwitchSectionObject *self,
                                     PyObject *value,
                                     void *closure) {
    SEQUENCE_TO_ARRAY(self->lambda_offsets, self->phase_count, (float) PyFloat_AsDouble, value)
}


int CalibrationSection_traverse(CalibrationSectionObject *self,
                                visitproc visit,
                                void *arg) {
    return 0;
}

int CalibrationSection_clear(CalibrationSectionObject *self) {
    return 0;
}

void CalibrationSection_dealloc(CalibrationSectionObject *self) {
    PyObject_GC_UnTrack(self);
    CalibrationSection_clear(self);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

int CalibrationSection_read(CalibrationSectionObject *self,
                            FILE *file) {
    fread_unlocked(&self->beam_efficiency,
                   sizeof(CalibrationSectionObject) - offsetof(CalibrationSectionObject, beam_efficiency), 1,
                   file);
    return 0;
}


PyMemberDef GeneralSection_members[] = {
    {
        .name = "ut",
        .type = Py_T_DOUBLE,
        .offset = offsetof(GeneralSectionObject, ut)
    },
    {
        .name = "lst",
        .type = Py_T_DOUBLE,
        .offset = offsetof(GeneralSectionObject, lst)
    },
    {
        .name = "azimuth",
        .type = Py_T_FLOAT,
        .offset = offsetof(GeneralSectionObject, azimuth)
    },
    {
        .name = "elevation",
        .type = Py_T_FLOAT,
        .offset = offsetof(GeneralSectionObject, elevation)
    },
    {
        .name = "opacity",
        .type = Py_T_FLOAT,
        .offset = offsetof(GeneralSectionObject, opacity)
    },
    {
        .name = "temperature",
        .type = Py_T_FLOAT,
        .offset = offsetof(GeneralSectionObject, temperature)
    },
    {
        .name = "integration_time",
        .type = Py_T_FLOAT,
        .offset = offsetof(GeneralSectionObject, integration_time)
    },
    {
        .name = "parallactic_angle",
        .type = Py_T_DOUBLE,
        .offset = offsetof(GeneralSectionObject, parallactic_angle)
    },
    {NULL}
};
PyTypeObject GeneralSectionType = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "GeneralSection",
    .tp_basicsize = sizeof(GeneralSectionObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
    .tp_traverse = (traverseproc) GeneralSection_traverse,
    .tp_clear = (inquiry) GeneralSection_clear,
    .tp_new = PyType_GenericNew,
    .tp_dealloc = (destructor) GeneralSection_dealloc,
    .tp_members = GeneralSection_members
};

PyMemberDef PositionSection_members[] = {
    {
        .name = "coordinate_system",
        .type = Py_T_INT,
        .offset = offsetof(PositionSectionObject, coordinate_system)
    },
    {
        .name = "equinox_year",
        .type = Py_T_FLOAT,
        .offset = offsetof(PositionSectionObject, equinox_year)
    },
    {
        .name = "projection_system",
        .type = Py_T_INT,
        .offset = offsetof(PositionSectionObject, projection_system)
    },
    {
        .name = "center_lambda",
        .type = Py_T_DOUBLE,
        .offset = offsetof(PositionSectionObject, center_lambda)
    },
    {
        .name = "center_beta",
        .type = Py_T_DOUBLE,
        .offset = offsetof(PositionSectionObject, center_beta)
    },
    {
        .name = "projection_angle",
        .type = Py_T_DOUBLE,
        .offset = offsetof(PositionSectionObject, projection_angle)
    },
    {
        .name = "lambda_offset",
        .type = Py_T_FLOAT,
        .offset = offsetof(PositionSectionObject, lambda_offset)
    },
    {
        .name = "beta_offset",
        .type = Py_T_FLOAT,
        .offset = offsetof(PositionSectionObject, beta_offset)
    },
    {NULL}
};
PyGetSetDef PositionSection_getset[] = {
    {
        .name = "source",
        .get = (getter) PositionSection_get_source,
        .set = (setter) PositionSection_set_source
    },
    {NULL}
};
PyTypeObject PositionSectionType = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "PositionSection",
    .tp_basicsize = sizeof(PositionSectionObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
    .tp_traverse = (traverseproc) PositionSection_traverse,
    .tp_clear = (inquiry) PositionSection_clear,
    .tp_new = PositionSection_new,
    .tp_dealloc = (destructor) PositionSection_dealloc,
    .tp_members = PositionSection_members,
    .tp_getset = PositionSection_getset
};

PyMemberDef SpectroSection_members[] = {
    {
        .name = "rest_frequency",
        .type = Py_T_DOUBLE,
        .offset = offsetof(SpectroSectionObject, rest_frequency)
    },
    {
        .name = "channel_count",
        .type = Py_T_INT,
        .offset = offsetof(SpectroSectionObject, channel_count)
    },
    {
        .name = "reference_channel",
        .type = Py_T_FLOAT,
        .offset = offsetof(SpectroSectionObject, reference_channel)
    },
    {
        .name = "frequency_resolution",
        .type = Py_T_FLOAT,
        .offset = offsetof(SpectroSectionObject, frequency_resolution)
    },
    {
        .name = "frequency_offset",
        .type = Py_T_FLOAT,
        .offset = offsetof(SpectroSectionObject, frequency_offset)
    },
    {
        .name = "velocity_resolution",
        .type = Py_T_FLOAT,
        .offset = offsetof(SpectroSectionObject, velocity_resolution)
    },
    {
        .name = "velocity_offset",
        .type = Py_T_FLOAT,
        .offset = offsetof(SpectroSectionObject, velocity_offset)
    },
    {
        .name = "image_frequency",
        .type = Py_T_DOUBLE,
        .offset = offsetof(SpectroSectionObject, image_frequency)
    },
    {
        .name = "velocity_type",
        .type = Py_T_INT,
        .offset = offsetof(SpectroSectionObject, velocity_type)
    },
    {
        .name = "doppler_correction",
        .type = Py_T_DOUBLE,
        .offset = offsetof(SpectroSectionObject, doppler_correction)
    },
    {NULL}
};
PyGetSetDef SpectroSection_getset[] = {
    {
        .name = "line",
        .get = (getter) SpectroSection_get_line,
        .set = (setter) SpectroSection_set_line
    },
    {NULL}
};
PyTypeObject SpectroSectionType = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "SpectroSection",
    .tp_basicsize = sizeof(SpectroSectionObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
    .tp_traverse = (traverseproc) SpectroSection_traverse,
    .tp_clear = (inquiry) SpectroSection_clear,
    .tp_new = SpectroSection_new,
    .tp_dealloc = (destructor) SpectroSection_dealloc,
    .tp_members = SpectroSection_members,
    .tp_getset = SpectroSection_getset
};

PyMemberDef PlotSection_members[] = {
    {
        .name = "intensity_min",
        .type = Py_T_FLOAT,
        .offset = offsetof(PlotSectionObject, intensity_min)
    },
    {
        .name = "intensity_max",
        .type = Py_T_FLOAT,
        .offset = offsetof(PlotSectionObject, intensity_max)
    },
    {
        .name = "velocity_min",
        .type = Py_T_FLOAT,
        .offset = offsetof(PlotSectionObject, velocity_min)
    },
    {
        .name = "velocity_max",
        .type = Py_T_FLOAT,
        .offset = offsetof(PlotSectionObject, velocity_max)
    },
    {NULL}
};
PyTypeObject PlotSectionType = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "PlotSection",
    .tp_basicsize = sizeof(PlotSectionObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
    .tp_traverse = (traverseproc) PlotSection_traverse,
    .tp_clear = (inquiry) PlotSection_clear,
    .tp_new = PyType_GenericNew,
    .tp_dealloc = (destructor) PlotSection_dealloc,
    .tp_members = PlotSection_members
};

PyMemberDef SwitchSection_members[] = {
    {
        .name = "switching_mode",
        .type = Py_T_INT,
        .offset = offsetof(SwitchSectionObject, switching_mode)
    },
    {NULL}
};
PyGetSetDef SwitchSection_getset[] = {
    {
        .name = "phase_count",
        .get = (getter) SwitchSection_get_phase_count,
        .set = (setter) SwitchSection_set_phase_count
    },
    {
        .name = "frequency_offsets",
        .get = (getter) SwitchSection_get_frequency_offsets,
        .set = (setter) SwitchSection_set_frequency_offsets
    },
    {
        .name = "times",
        .get = (getter) SwitchSection_get_times,
        .set = (setter) SwitchSection_set_times
    },
    {
        .name = "weights",
        .get = (getter) SwitchSection_get_weights,
        .set = (setter) SwitchSection_set_weights
    },
    {
        .name = "lambda_offsets",
        .get = (getter) SwitchSection_get_lambda_offsets,
        .set = (setter) SwitchSection_set_lambda_offsets
    },
    {
        .name = "beta_offsets",
        .get = (getter) SwitchSection_get_beta_offsets,
        .set = (setter) SwitchSection_set_beta_offsets
    },
    {NULL}
};
PyTypeObject SwitchSectionType = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "SwitchSection",
    .tp_basicsize = sizeof(SwitchSectionObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
    .tp_traverse = (traverseproc) SwitchSection_traverse,
    .tp_clear = (inquiry) SwitchSection_clear,
    .tp_new = PyType_GenericNew,
    .tp_dealloc = (destructor) SwitchSection_dealloc,
    .tp_members = SwitchSection_members,
    .tp_getset = SwitchSection_getset,
};

PyMemberDef CalibrationSection_members[] = {
    {
        .name = "beam_efficiency",
        .type = Py_T_FLOAT,
        .offset = offsetof(CalibrationSectionObject, beam_efficiency)
    },
    {
        .name = "forward_efficiency",
        .type = Py_T_FLOAT,
        .offset = offsetof(CalibrationSectionObject, forward_efficiency)
    },
    {
        .name = "gain_ratio",
        .type = Py_T_FLOAT,
        .offset = offsetof(CalibrationSectionObject, gain_ratio)
    },
    {
        .name = "water_content",
        .type = Py_T_FLOAT,
        .offset = offsetof(CalibrationSectionObject, water_content)
    },
    {
        .name = "ambient_pressure",
        .type = Py_T_FLOAT,
        .offset = offsetof(CalibrationSectionObject, ambient_pressure)
    },
    {
        .name = "ambient_temperature",
        .type = Py_T_FLOAT,
        .offset = offsetof(CalibrationSectionObject, ambient_temperature)
    },
    {
        .name = "signal_atmosphere_temperature",
        .type = Py_T_FLOAT,
        .offset = offsetof(CalibrationSectionObject, signal_atmosphere_temperature)
    },
    {
        .name = "chopper_temperature",
        .type = Py_T_FLOAT,
        .offset = offsetof(CalibrationSectionObject, chopper_temperature)
    },
    {
        .name = "cold_load_temperature",
        .type = Py_T_FLOAT,
        .offset = offsetof(CalibrationSectionObject, cold_load_temperature)
    },
    {
        .name = "signal_opacity",
        .type = Py_T_FLOAT,
        .offset = offsetof(CalibrationSectionObject, signal_opacity)
    },
    {
        .name = "image_opacity",
        .type = Py_T_FLOAT,
        .offset = offsetof(CalibrationSectionObject, image_opacity)
    },
    {
        .name = "image_atmosphere_temperature",
        .type = Py_T_FLOAT,
        .offset = offsetof(CalibrationSectionObject, image_atmosphere_temperature)
    },
    {
        .name = "receiver_temperature",
        .type = Py_T_FLOAT,
        .offset = offsetof(CalibrationSectionObject, receiver_temperature)
    },
    {
        .name = "mode",
        .type = Py_T_INT,
        .offset = offsetof(CalibrationSectionObject, mode)
    },
    {
        .name = "factor",
        .type = Py_T_FLOAT,
        .offset = offsetof(CalibrationSectionObject, factor)
    },
    {
        .name = "site_elevation",
        .type = Py_T_FLOAT,
        .offset = offsetof(CalibrationSectionObject, site_elevation)
    },
    {
        .name = "atmosphere_power",
        .type = Py_T_FLOAT,
        .offset = offsetof(CalibrationSectionObject, atmosphere_power)
    },
    {
        .name = "chopper_power",
        .type = Py_T_FLOAT,
        .offset = offsetof(CalibrationSectionObject, chopper_power)
    },
    {
        .name = "cold_power",
        .type = Py_T_FLOAT,
        .offset = offsetof(CalibrationSectionObject, cold_power)
    },
    {
        .name = "longitude_offset",
        .type = Py_T_FLOAT,
        .offset = offsetof(CalibrationSectionObject, longitude_offset)
    },
    {
        .name = "latitude_offset",
        .type = Py_T_FLOAT,
        .offset = offsetof(CalibrationSectionObject, latitude_offset)
    },
    {
        .name = "geographic_longitude",
        .type = Py_T_DOUBLE,
        .offset = offsetof(CalibrationSectionObject, geographic_longitude)
    },
    {
        .name = "geographic_latitude",
        .type = Py_T_DOUBLE,
        .offset = offsetof(CalibrationSectionObject, geographic_latitude)
    },
    {NULL}
};
PyTypeObject CalibrationSectionType = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "CalibrationSection",
    .tp_basicsize = sizeof(CalibrationSectionObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
    .tp_traverse = (traverseproc) CalibrationSection_traverse,
    .tp_clear = (inquiry) CalibrationSection_clear,
    .tp_new = PyType_GenericNew,
    .tp_dealloc = (destructor) CalibrationSection_dealloc,
    .tp_members = CalibrationSection_members
};
