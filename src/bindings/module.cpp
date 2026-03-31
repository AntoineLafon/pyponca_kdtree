#include "module.hpp"

// utils functions
inline py::array_t<int> collect( auto& query, size_t reserve = 0 ){
    std::vector<int> values;
    if( reserve > 0 ){
        values.reserve(reserve);
    }

    for( const auto neighbor : query ){
        values.push_back(neighbor);
    }

    py::array_t<int> out(values.size());
    auto view = out.mutable_unchecked<1>();
    for( size_t i = 0; i < values.size(); ++i ){
        view(i) = values[i];
    }
    return out;
}


void verifyQueryPoint( const py::array& query ){
    if( query.ndim() != 1 || query.shape(0) != 3 ){
        throw py::value_error("query must have shape (3,)");
    }
}

void verifyPointcloud( const py::array& points ){
    if( points.ndim() != 2 || points.shape(1) != 3 ){
        throw py::value_error("points must have shape (N, 3)");
    }   
}

void verifyIndex( size_t index, const std::vector<PoncaPoint>& points ){
    if( index >= points.size() ){
        throw py::value_error("index out of bounds");
    }
}

//KDTree implementation
KdTree::KdTree( const py::array_t<float>& points ){

    verifyPointcloud(points);  

    const size_t size = points.shape(0);
    m_points.reserve(size);

    auto uncheckedPoints = points.unchecked<2>();

    for( int i = 0; i < size; i++ ){
        PoncaPoint pts;
        for( int j = 0; j < 3; j++ ){
            pts.pos()[j] = uncheckedPoints(i, j);
        }
        m_points.emplace_back(pts);
    }

    m_tree.build(m_points);

}

std::size_t KdTree::size() const {
    return m_points.size();
}

py::array_t<int> KdTree::query_knn( const py::array_t<float>& query, int k ) const {
    verifyQueryPoint(query);

    VectorType queryPoint;
    auto uncheckedQuery = query.unchecked<1>();
    for( int j = 0; j < 3; j++ ){
        queryPoint(j) = uncheckedQuery(j);
    }

    auto neighbors = m_tree.kNearestNeighbors(queryPoint, k);
    return collect(neighbors, static_cast<size_t>(k));
}


py::array_t<int> KdTree::query_radius( const py::array_t<float>& query, float radius ) const {
    verifyQueryPoint(query);

    VectorType queryPoint;
    auto uncheckedQuery = query.unchecked<1>();
    for( int j = 0; j < 3; j++ ){
        queryPoint(j) = uncheckedQuery(j);
    }

    auto neighbors = m_tree.rangeNeighbors(queryPoint, radius);
    return collect(neighbors);
}

py::array_t<int> KdTree::query_knn_index( int index, int k ) const {
    verifyIndex(index, m_points);

    auto neighbors = m_tree.kNearestNeighbors(static_cast<int>(index), static_cast<int>(k));
    return collect(neighbors, static_cast<size_t>(k));
}

py::array_t<int> KdTree::query_radius_index( int index, float radius ) const {
    verifyIndex(index, m_points);

    auto neighbors = m_tree.rangeNeighbors(static_cast<int>(index), radius);
    return collect(neighbors);
}


//KNN graph implementation
KnnGraph::KnnGraph(const py::array_t<float>& points, int k){
    verifyPointcloud(points);  

    const size_t size = points.shape(0);
    m_points.reserve(size);

    auto uncheckedPoints = points.unchecked<2>();

    for( int i = 0; i < size; i++ ){
        PoncaPoint pts;
        for( int j = 0; j < 3; j++ ){
            pts.pos()[j] = uncheckedPoints(i, j);
        }
        m_points.emplace_back(pts);
    }

    m_kdtree.build(m_points);

    m_graph = std::make_unique<PoncaKnnGraph>(m_kdtree, k);
}

std::size_t KnnGraph::size() const {
    return m_points.size();
}

py::array_t<int> KnnGraph::query_knn_index(int index) const {
    verifyIndex(index, m_points);

    auto neighbors = m_graph->kNearestNeighbors(static_cast<int>(index));
    return collect(neighbors, static_cast<size_t>(m_graph->k()));
}


py::array_t<int> KnnGraph::query_radius_index(int index, float radius) const {
    verifyIndex(index, m_points);

    auto neighbors = m_graph->rangeNeighbors(static_cast<int>(index), radius);
    return collect(neighbors);
}
