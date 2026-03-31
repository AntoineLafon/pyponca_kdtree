#include "module.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

PYBIND11_MODULE(ponca_kdtree, m)
{
    py::class_<KdTree>(m, "KdTree")
        .def(py::init<const py::array&>(), py::arg("points"))
        .def_property_readonly("size", &KdTree::size)
        .def("query_knn", &KdTree::query_knn, py::arg("query"), py::arg("k"))
        .def("query_radius", &KdTree::query_radius, py::arg("query"), py::arg("radius"))
        .def("query_knn_index", &KdTree::query_knn_index, py::arg("index"), py::arg("k"))
        .def("query_radius_index", &KdTree::query_radius_index, py::arg("index"), py::arg("radius"));
    
    py::class_<KnnGraph>(m, "KnnGraph")
        .def(py::init<const py::array&, int>(), py::arg("points"), py::arg("graph_k"))
        .def_property_readonly("size", &KnnGraph::size)
        .def("query_knn_index", &KnnGraph::query_knn_index, py::arg("index"))
        .def("query_radius_index", &KnnGraph::query_radius_index, py::arg("index"), py::arg("radius"));
}