#pragma once

#include <Python.h>

#define GENERAL_SECTION_ID -2
#define POSITION_SECTION_ID -3
#define SPECTROSCOPIC_SECTION_ID -4
#define DEFAULT_PLOTTING_SECTION_ID -7
#define SWITCHING_SECTION_ID -8
#define CALIBRATION_SECTION_ID -14


#pragma pack(1)

typedef struct GeneralSectionObject {
    PyObject_HEAD
    double ut;
    double lst;
    float azimuth;
    float elevation;
    float opacity;
    float temperature;
    float integration_time;
    double parallactic_angle;
} GeneralSectionObject;

#pragma pack()
