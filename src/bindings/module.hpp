#pragma once

#include <type_traits>
#include <utility>

#include <Ponca/src/Common/pointTypes.h>
#include <Ponca/src/SpatialPartitioning/KdTree/kdTree.h>
#include <Ponca/src/SpatialPartitioning/KnnGraph/knnGraph.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

namespace py = pybind11;

using PoncaPoint = Ponca::PointPosition<float, 3>;
using VectorType = PoncaPoint::VectorType;
using PoncaKdTree = Ponca::KdTreeDense<PoncaPoint>;
using PoncaKnnGraph = Ponca::KnnGraph<PoncaPoint>;


class KdTree{

    public:
    
        KdTree( const py::array_t<float>& points );

        std::size_t size() const;

        py::array_t<int> query_knn( const py::array_t<float>& query, int k ) const;
        py::array_t<int> query_radius( const py::array_t<float>& query, float radius ) const;

        py::array_t<int> query_knn_index( int index, int k ) const;
        py::array_t<int> query_radius_index( int index, float radius ) const;
    
    private:

        std::vector<PoncaPoint> m_points;
        PoncaKdTree m_tree;

};


class KnnGraph{

    public:
        KnnGraph(const py::array_t<float>& points, int k);

        std::size_t size() const;

        py::array_t<int> query_knn_index(int index) const;
        py::array_t<int> query_radius_index(int index, float radius) const;

    private:

        std::vector<PoncaPoint> m_points;
        PoncaKdTree m_kdtree;
        std::unique_ptr<PoncaKnnGraph> m_graph;

};
