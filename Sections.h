#pragma once

#include <Python.h>

#include "Utils.h"

#define GENERAL_SECTION_ID -2
#define POSITION_SECTION_ID -3
#define SPECTRO_SECTION_ID -4
#define PLOT_SECTION_ID -7
#define SWITCH_SECTION_ID -8
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

    char source[3 * WORD_SIZE];
    int coordinate_system; // code
    float equinox;
    int projection_system; // code
    double center_lambda;
    double center_beta;
    double projection_angle;
    float lambda_offset;
    float beta_offset;
} PositionSectionObject;

typedef struct SpectroSectionObject {
    PyObject_HEAD

    char line[3 * WORD_SIZE];
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
} SpectroSectionObject;

typedef struct PlotSectionObject {
    PyObject_HEAD

    float intensity_min;
    float intensity_max;
    float velocity_min;
    float velocity_max;
} PlotSectionObject;

typedef struct SwitchSectionObject {
    PyObject_HEAD

    int phase_count;
    double *frequency_offsets;
    float *times;
    float *weights;
    int mode; // code
    float *lambda_offsets;
    float *beta_offsets;
} SwitchSectionObject;

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

typedef int (*section_read)(PyObject *, FILE *);

section_read get_section_read(int id);


int GeneralSection_traverse(GeneralSectionObject *self,
                            visitproc visit,
                            void *arg);

int GeneralSection_clear(GeneralSectionObject *self);

void GeneralSection_dealloc(GeneralSectionObject *self);

int GeneralSection_read(GeneralSectionObject *self,
                        FILE *file);


int PositionSection_traverse(PositionSectionObject *self,
                             visitproc visit,
                             void *arg);

int PositionSection_clear(PositionSectionObject *self);

void PositionSection_dealloc(PositionSectionObject *self);

PyObject *PositionSection_new(PyTypeObject *type,
                              PyObject *args,
                              PyObject *kwds);

int PositionSection_read(PositionSectionObject *self,
                         FILE *file);

PyObject *PositionSection_get_source(const PositionSectionObject *self,
                                     void *closure);

int PositionSection_set_source(PositionSectionObject *self,
                               PyObject *value,
                               void *closure);


int SpectroSection_traverse(SpectroSectionObject *self,
                            visitproc visit,
                            void *arg);

int SpectroSection_clear(SpectroSectionObject *self);

void SpectroSection_dealloc(SpectroSectionObject *self);

PyObject *SpectroSection_new(PyTypeObject *type,
                             PyObject *args,
                             PyObject *kwds);

int SpectroSection_read(SpectroSectionObject *self,
                        FILE *file);

PyObject *SpectroSection_get_line(const SpectroSectionObject *self,
                                  void *closure);

int SpectroSection_set_line(SpectroSectionObject *self,
                            PyObject *value,
                            void *closure);


int PlotSection_traverse(PlotSectionObject *self,
                         visitproc visit,
                         void *arg);

int PlotSection_clear(PlotSectionObject *self);

void PlotSection_dealloc(PlotSectionObject *self);

int PlotSection_read(PlotSectionObject *self,
                     FILE *file);


int SwitchSection_traverse(SwitchSectionObject *self,
                           visitproc visit,
                           void *arg);

int SwitchSection_clear(SwitchSectionObject *self);

void SwitchSection_dealloc(SwitchSectionObject *self);

int SwitchSection_read(SwitchSectionObject *self,
                       FILE *file);

int SwitchSection_alloc_arrays(SwitchSectionObject *self);

PyObject *SwitchSection_get_phase_count(const SwitchSectionObject *self,
                                        void *closure);

int SwitchSection_set_phase_count(SwitchSectionObject *self,
                                  PyObject *value,
                                  void *closure);

PyObject *SwitchSection_get_frequency_offsets(const SwitchSectionObject *self,
                                              void *closure);

int SwitchSection_set_frequency_offsets(SwitchSectionObject *self,
                                        PyObject *value,
                                        void *closure);

PyObject *SwitchSection_get_times(const SwitchSectionObject *self,
                                  void *closure);

int SwitchSection_set_times(SwitchSectionObject *self,
                            PyObject *value,
                            void *closure);

PyObject *SwitchSection_get_weights(const SwitchSectionObject *self,
                                    void *closure);

int SwitchSection_set_weights(SwitchSectionObject *self,
                              PyObject *value,
                              void *closure);

PyObject *SwitchSection_get_beta_offsets(const SwitchSectionObject *self,
                                         void *closure);

int SwitchSection_set_beta_offsets(SwitchSectionObject *self,
                                   PyObject *value,
                                   void *closure);

PyObject *SwitchSection_get_lambda_offsets(const SwitchSectionObject *self,
                                           void *closure);

int SwitchSection_set_lambda_offsets(SwitchSectionObject *self,
                                     PyObject *value,
                                     void *closure);


int CalibrationSection_traverse(CalibrationSectionObject *self,
                                visitproc visit,
                                void *arg);

int CalibrationSection_clear(CalibrationSectionObject *self);

void CalibrationSection_dealloc(CalibrationSectionObject *self);

int CalibrationSection_read(CalibrationSectionObject *self,
                            FILE *file);


extern PyMemberDef GeneralSection_members[];
extern PyTypeObject GeneralSectionType;

extern PyMemberDef PositionSection_members[];
extern PyGetSetDef PositionSection_getset[];
extern PyTypeObject PositionSectionType;

extern PyMemberDef SpectroSection_members[];
extern PyGetSetDef SpectroSection_getset[];
extern PyTypeObject SpectroSectionType;

extern PyMemberDef PlotSection_members[];
extern PyTypeObject PlotSectionType;

extern PyMemberDef SwitchSection_members[];
extern PyGetSetDef SwitchSection_getset[];
extern PyTypeObject SwitchSectionType;

extern PyMemberDef CalibrationSection_members[];
extern PyTypeObject CalibrationSectionType;
