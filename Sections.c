#include "Sections.h"


PyObject *GeneralSection_traverse(GeneralSectionObject *self,
                                  visitproc visit,
                                  void *arg) {
    return 0;
}

PyObject *GeneralSection_clear(GeneralSectionObject *self) {

}

void GeneralSection_dealloc(GeneralSectionObject *self);


PyObject *PositionSection_traverse(GeneralSectionObject *self,
                                   visitproc visit,
                                   void *arg);

PyObject *PositionSection_clear(GeneralSectionObject *self);

void PositionSection_dealloc(GeneralSectionObject *self);

PyObject *PositionSection_get_source(const PositionSectionObject *self,
                                     void *closure);

PyObject *PositionSection_set_source(const PositionSectionObject *self,
                                     PyObject *value,
                                     void *closure);


PyObject *SpectroSection_traverse(GeneralSectionObject *self,
                                  visitproc visit,
                                  void *arg);

PyObject *SpectroSection_clear(GeneralSectionObject *self);

void SpectroSection_dealloc(GeneralSectionObject *self);

PyObject *SpectroSection_get_line(const SpectroSectionObject *self,
                                  void *closure);

PyObject *SpectroSection_set_line(const SpectroSectionObject *self,
                                  PyObject *value,
                                  void *closure);


PyObject *PlotSection_traverse(GeneralSectionObject *self,
                               visitproc visit,
                               void *arg);

PyObject *PlotSection_clear(GeneralSectionObject *self);

void PlotSection_dealloc(GeneralSectionObject *self);


PyObject *SwitchSection_traverse(GeneralSectionObject *self,
                                 visitproc visit,
                                 void *arg);

PyObject *SwitchSection_clear(GeneralSectionObject *self);

void SwitchSection_dealloc(GeneralSectionObject *self);


PyObject *CalibrationSection_traverse(GeneralSectionObject *self,
                                      visitproc visit,
                                      void *arg);

PyObject *CalibrationSection_clear(GeneralSectionObject *self);

void CalibrationSection_dealloc(GeneralSectionObject *self);


PyMemberDef GeneralSection_members[];
PyTypeObject GeneralSectionType;

PyMemberDef PositionSection_members[];
PyGetSetDef PositionSection_getset[];
PyTypeObject PositionSectionType;

PyMemberDef SpectroSection_members[];
PyGetSetDef SpectroSection_getset[];
PyTypeObject SpectroSectionType;

PyMemberDef PlottingSection_members[];
PyTypeObject PlottingSectionType;

PyMemberDef SwitchingSection_members[];
PyTypeObject SwitchingSectionType;

PyMemberDef CalibrationSection_members[];
PyTypeObject CalibrationSectionType;
