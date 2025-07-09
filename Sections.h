#pragma once

#include <Python.h>

#include "Utils.h"

#define GENERAL_SECTION_ID -2
#define POSITION_SECTION_ID -3
#define SPECTRO_SECTION_ID -4
#define PLOTTING_SECTION_ID -7
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

typedef struct PositionSectionObject {
    PyObject_HEAD

    char source[3 * WORD_LENGTH];
    int coordinate_system; // code
    float equinox_year;
    int projection_system; // code
    double center_lambda;
    double center_beta;
    double projection_angle;
    float lambda_offset;
    float beta_offset;
} PositionSectionObject;

typedef struct SpectroSectionObject {
    PyObject_HEAD

    char line[3 * WORD_LENGTH];
    double rest_frequency;
    int channel_count;
    float reference_channel;
    float frequency_resolution;
    float frequency_offset;
    float velocity_resolution;
    float velocity_offset;
    float blank;
    double image_frequency;
    int velocity_type; // code
    double doppler_correction;
} SpectroscopicObject;

typedef struct PlottingSectionObject {
    PyObject_HEAD

    float intensity_min;
    float intensity_max;
    float velocity_min;
    float velocity_max;
} DefaultPlottingObject;

typedef struct SwitchingSectionObject {
    PyObject_HEAD

    int phase_count;
    double *frequency_offsets;
    float *times;
    float *weights;
    int switching_mode; // code
    float *lambda_offsets;
    float *beta_offsets;
} SwitchingSectionObject;

typedef struct CalibrationSectionObject {
    PyObject_HEAD

    float beam_efficiency;
    float forward_efficiency;
    float gain_ratio;
    float water_content;
    float ambient_pressure;
    float ambient_temperature;
    float signal_atmosphere_temperature;
    float chopper_temperature;
    float cold_load_temperature;
    float signal_opacity;
    float image_opacity;
    float image_atmosphere_temperature;
    float receiver_temperature;
    int mode; // code
    float factor;
    float site_elevation;
    float atmosphere_power;
    float chopper_power;
    float cold_power;
    float longitude_offset;
    float latitude_offset;
    double geographic_longitude;
    double geographic_latitude;
} CalibrationSectionObject;

#pragma pack()

extern PyMemberDef GeneralSection_members[];
extern PyTypeObject GeneralSectionType;

extern PyMemberDef PositionSection_members[];
extern PyGetSetDef PositionSection_getset[];
extern PyTypeObject PositionSectionType;

extern PyMemberDef SpectroSection_members[];
extern PyGetSetDef SpectroSection_getset[];
extern PyTypeObject SpectroSectionType;

extern PyMemberDef PlottingSection_members[];
extern PyTypeObject PlottingSectionType;

extern PyMemberDef SwitchingSection_members[];
extern PyTypeObject SwitchingSectionType;

extern PyMemberDef CalibrationSection_members[];
extern PyTypeObject CalibrationSectionType;
