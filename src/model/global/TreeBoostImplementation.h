#ifndef TREEBOOSTIMPLEMENTATION_H
#define TREEBOOSTIMPLEMENTATION_H

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/reverse_graph.hpp>
#include <boost/graph/graphml.hpp>

namespace faceAnalysis {
  namespace {
    using std::map;
    using cv::Mat;

    map<E, Mat> graph_covariances_vector;
    map<E, Mat> graph_covariances_vector_inverted;
    map<E, Mat> graph_mean_vector;
    map<E, Mat> graph_svd_u_vector;
    map<E, Mat> graph_svd_d_vector;
    map<E, double> graph_weight_vector;

    std::vector<E> tree_edges;
    map<E, Mat> tree_covariances_vector;
    map<E, Mat> tree_covariances_vector_inverted;
    map<E, Mat> tree_mean_mat_vector;
    map<E, cv::Point2d> tree_mean_point_vector;
    map<E, Mat> tree_svd_u_vector;
    map<E, Mat> tree_svd_d_vector;
    map<E, double> tree_weight_vector;

    std::vector<int> used_verticies;
    std::vector<TreeVertex> leaves;
    bool leaves_search_init = true;

  }

  class FindTreeVisitor : public boost::default_dfs_visitor  {
  public:
    template < typename Vertex, typename Graph >
    void discover_vertex(Vertex u, const Graph& g) const {
      UNUSED(g);
      used_verticies.push_back(u);
    }

    template < typename Edge, typename Graph >
    void examine_edge(Edge e, const Graph& g) const {
      if (!(std::find(used_verticies.begin(), used_verticies.end(),
                      target(e, g)) != used_verticies.end()))
        {
          E directed_edge(source(e, g), target(e, g));
          tree_edges.push_back(directed_edge);

          tree_covariances_vector[directed_edge] =
              graph_covariances_vector[directed_edge];
          tree_covariances_vector_inverted[directed_edge] =
              graph_covariances_vector_inverted[directed_edge];
          tree_mean_mat_vector[directed_edge] =
              graph_mean_vector[directed_edge];
          tree_mean_point_vector[directed_edge] =
              cv::Point2d(graph_mean_vector[directed_edge].at<double>(0, 0),
                          graph_mean_vector[directed_edge].at<double>(1, 0));

          tree_svd_u_vector[directed_edge] = graph_svd_u_vector[directed_edge];
          tree_svd_d_vector[directed_edge] = graph_svd_d_vector[directed_edge];
          tree_weight_vector[directed_edge] =
              graph_weight_vector[directed_edge];
        }
    }
  };

  class FindLeafVisitor : public boost::default_dfs_visitor {
  public:
    template < typename Vertex, typename Tree >
    void start_vertex(Vertex v, const Tree& g) const {
      UNUSED(v);
      UNUSED(g);
      if (leaves_search_init)
        {
          leaves.clear();
          leaves_search_init = false;
        }
    }

    template < typename Edge, typename Tree >
    void examine_edge(Edge e, const Tree& g) const {
      UNUSED(e);
      UNUSED(g);
      TreeVertex vertex = target(e, g);
      if (out_degree(vertex, g) == 0)
        leaves.push_back(vertex);
    }
  };
}



#endif // TREEBOOSTIMPLEMENTATION_H


