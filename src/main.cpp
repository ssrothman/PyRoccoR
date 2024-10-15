#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "roccor/RoccoR.h"

namespace py = pybind11;

PYBIND11_MODULE(_roccor, m) {
    m.doc() = R"pbdoc(
        placeholder
    )pbdoc";

    py::class_<RoccoR>(m, "RoccoR")
        .def(py::init<>())
        .def(py::init<const std::string &>())
        .def("init",           &RoccoR::init)
        .def("reset",          &RoccoR::reset)
        .def("kScalDTerror",   py::vectorize(&RoccoR::kScaleDTerror))
        .def("kSpreadMCerror", py::vectorize(&RoccoR::kSpreadMCerror))
        .def("kSmearMCerror",  py::vectorize(&RoccoR::kSmearMCerror))
        .def("kScaleDT",       py::vectorize(&RoccoR::kScaleDT))
        .def("kScaleMC",       py::vectorize(&RoccoR::kScaleMC))
        .def("kSpreadMC",      py::vectorize(&RoccoR::kSpreadMC))
        .def("kSmearMC",       py::vectorize(&RoccoR::kSmearMC));

}
