#include "GlobalTree.h"

#include <opencv2/highgui/highgui.hpp>

#include "TreeBoostImplementation.h"

log4cxx::LoggerPtr faceAnalysis::GlobalTree::logger(
    log4cxx::Logger::getLogger("CPT.models.training.globalTree"));

std::map<int, cv::Point2d> faceAnalysis::GlobalTree::all_means(
    const int rootID, const cv::Point2d root) const {

  std::map<int, cv::Point2d> mean_positions;
  mean_positions[rootID] = root;
  for (std::vector<E>::iterator itr = tree_edges.begin();
       itr != tree_edges.end(); ++itr)
    {
      E edge = *itr;
      mean_positions[itr->second] = mean_positions[itr->first]
          - tree_mean_point_vector[edge];
    }

  return mean_positions;
}


void faceAnalysis::GlobalTree::set_current_edge(const E &current_edge)
{
  current_edge_ = current_edge;
  current_edge_mean_ = tree_mean_mat_vector[current_edge];
  current_edge_cov_inv_ = tree_covariances_vector_inverted[current_edge];
}


void faceAnalysis::GlobalTree::Train(const Dataset data) {
  using cv::Mat;
  using std::vector;

  int image_counter = 0;
  trainDataSize_ = data.size();

  Mat dataset_mat;
  dataset_mat.create(data.size(), data.at(0).second.rows * 2, CV_64F);
  dataset_mat = cv::Scalar(0);

  for (Dataset::const_iterator itr = data.begin(); itr != data.end(); ++itr)
    {
      DatasetExample sample = *itr;
      PointsOfImage ground_truth_points = sample.second;
      ground_truth_points = ground_truth_points.reshape(0, 1);
      ground_truth_points.copyTo(dataset_mat.row(image_counter));
      image_counter++;
    }

  // Total connection - edges
  vector<E> full_connection_edges;
  for (int i = 0; i < 2 * data.at(0).second.rows - 1; i = i + 2)
    {
      for (int j = i + 2; j <= 2 * data.at(0).second.rows - 1; j = j + 2)
        {
          full_connection_edges.push_back(E(i / 2, j / 2));
          full_connection_edges.push_back(E(j / 2, i / 2));
        }
    }

  // calculate edge values
  for (uint edge_count = 0; edge_count < full_connection_edges.size();
       edge_count++)
    {
      E edge = full_connection_edges.at(edge_count);

      // dirty hack
      bool edge_source = find(landmarks_to_model_.begin(),
                              landmarks_to_model_.end(),
                              edge.first) != landmarks_to_model_.end();
      bool edge_target = find(landmarks_to_model_.begin(),
                              landmarks_to_model_.end(),
                              edge.second) != landmarks_to_model_.end();

      if (edge_source && edge_target)
        {
          Mat l_i(data.size(), 2, CV_64F);
          l_i = cv::Scalar(0);

          Mat l_j(data.size(), 2, CV_64F);
          l_j = cv::Scalar(0);

          Mat l_i_x = dataset_mat.col(edge.first * 2);
          Mat l_i_y = dataset_mat.col(edge.first * 2 + 1);

          Mat l_j_x = dataset_mat.col(edge.second * 2);
          Mat l_j_y = dataset_mat.col(edge.second * 2 + 1);

          l_i_x.copyTo(l_i.col(0));
          l_i_y.copyTo(l_i.col(1));
          l_j_x.copyTo(l_j.col(0));
          l_j_y.copyTo(l_j.col(1));

          Mat diff = l_i - l_j;

          Mat covariance;
          Mat mean;
          calcCovarMatrix(diff, covariance, mean,
                          CV_COVAR_SCALE | CV_COVAR_NORMAL | CV_COVAR_ROWS);

          graph_covariances_vector[edge] = covariance;
          graph_covariances_vector_inverted[edge] = covariance.inv();
          graph_mean_vector[edge] = mean.t();

          cv::SVD svd(covariance);
          Mat u = svd.u;
          Mat d = Mat::diag(svd.w);

          graph_svd_u_vector[edge] = u;
          graph_svd_d_vector[edge] = d;

          double traceOfCov = trace(covariance)(0);
          double mse = traceOfCov;

          graph_weight_vector[edge] = mse;
        }
    }

  // create total connected undirected graph
  Graph total_connected_graph;
  boost::property_map<Graph, boost::edge_weight_t>::type weight_map =
      get(boost::edge_weight, total_connected_graph);

  map<E, double>::iterator weight_iterator;
  for (weight_iterator = graph_weight_vector.begin();
       weight_iterator != graph_weight_vector.end(); weight_iterator++)
    {
      E edge = weight_iterator->first;
      GraphEdge e;
      bool inserted;
      tie(e, inserted) = add_edge(edge.first, edge.second, total_connected_graph);
      weight_map[e] = graph_weight_vector[edge];
    }

  // calcuate undirected minimal spanning graph
  std::vector < GraphEdge > undirected_spanning_tree_edges;
  kruskal_minimum_spanning_tree(
        total_connected_graph, std::back_inserter(
          undirected_spanning_tree_edges));

  Graph undirected_spanning_tree;
  boost::property_map<Graph, boost::edge_weight_t>::type
      weightmap_undirected_tree = get(boost::edge_weight,
                                      undirected_spanning_tree);

  for (std::vector < GraphEdge >::iterator ei =
       undirected_spanning_tree_edges.begin();
       ei != undirected_spanning_tree_edges.end(); ++ei)
    {
      int source_idx = source(*ei, total_connected_graph);
      int target_idx = target(*ei, total_connected_graph);
      E edge = E(source_idx, target_idx);

      GraphEdge e;
      bool inserted;
      tie(e, inserted) = add_edge(source_idx, target_idx,
                                  undirected_spanning_tree);
      weightmap_undirected_tree[e] = graph_weight_vector[edge];
    }

  // get the root vertex
  GraphVertex rootVertex;
  rootVertex = source(undirected_spanning_tree_edges.at(0),
                      undirected_spanning_tree);


  // create directed minimal spanning tree
  tree_edges.clear();
  tree_covariances_vector.clear();
  tree_covariances_vector_inverted.clear();
  tree_mean_mat_vector.clear();
  tree_mean_point_vector.clear();
  tree_svd_u_vector.clear();
  tree_svd_d_vector.clear();
  tree_weight_vector.clear();
  used_verticies.clear();
  FindTreeVisitor find_tree_visitor;
  depth_first_search(undirected_spanning_tree,
                     visitor(find_tree_visitor).root_vertex(rootVertex));

  bool is_first_vertex = true;
  boost::property_map<Tree, boost::edge_weight_t>::type
      weightmap_directed_tree = get(boost::edge_weight, directedSpanningTree_);
  for (std::vector<E>::iterator ei = tree_edges.begin(); ei != tree_edges.end();
       ++ei)
    {
      int source_idx = ei->first;
      int target_idx = ei->second;

      LOG4CXX_INFO(logger, "Directed tree: " << source_idx << " --> "
                   << target_idx);
      TreeEdge e;
      bool inserted;
      tie(e, inserted) = add_edge(source_idx, target_idx, directedSpanningTree_);

      if (is_first_vertex)
        {
          treeRoot_ = source(e, directedSpanningTree_);
          is_first_vertex = false;
        }

      weightmap_directed_tree[e] = tree_weight_vector[*ei];
    }
}


void faceAnalysis::GlobalTree::PrintOutput(const std::string path) {
  // output tree
  std::ofstream output_file(path.c_str());
  output_file << "digraph A {\n" << " rankdir=TB\n" << " ratio=\"filled\"\n"
              << " edge[style=\"bold\"]\n" << " node[shape=\"circle\"]\n";

  boost::graph_traits<Tree>::edge_iterator eiter, eiter_end;
  for (tie(eiter, eiter_end) = edges(directedSpanningTree_);
       eiter != eiter_end; ++eiter)
    {
      output_file << source(*eiter, directedSpanningTree_) << " -> "
                  << target(*eiter, directedSpanningTree_);
      output_file << "[color=\"black\", label=\""
                  << get(boost::edge_weight, directedSpanningTree_, *eiter)
                  << "\"];\n";
    }
  output_file << "}\n";
}



void faceAnalysis::GlobalTree::Save(const std::string path) {
  LOG4CXX_INFO(logger, "Saving global tree");
  cv::FileStorage file(path, cv::FileStorage::WRITE);

  if (file.isOpened())
    {
      file << "trainDatacv::Size" << trainDataSize_;
      file << "edges" << "[";
      for (std::vector <E>::iterator ei = tree_edges.begin();
           ei != tree_edges.end(); ++ei)
        {
          file << "{" << "sourceIdx" <<  ei->first << "targetIdx" << ei->second
               << "weight" <<  tree_weight_vector.at(*ei)
               << "cov" << tree_covariances_vector.at(*ei)
               << "covInv" << tree_covariances_vector_inverted.at(*ei)
               << "sVec" << tree_mean_mat_vector.at(*ei)
               << "dVec" << tree_svd_d_vector.at(*ei)
               << "uVec" << tree_svd_u_vector.at(*ei) << "}";
        }

      file << "]";
      file.release();
    }
}

void faceAnalysis::GlobalTree::Load(const std::string path) {
  LOG4CXX_INFO(logger, "Loading global tree from " << path);
  directedSpanningTree_.clear();
  tree_edges.clear();
  tree_weight_vector.clear();
  tree_covariances_vector.clear();
  tree_mean_mat_vector.clear();
  tree_mean_point_vector.clear();
  tree_svd_d_vector.clear();
  tree_svd_u_vector.clear();
  leaves.clear();
  leaves_search_init = true;

  map<int, bool> verticies_count_map;

  boost::property_map<Tree, boost::edge_weight_t>::type
      weightmap_directed_tree = get(boost::edge_weight, directedSpanningTree_);

  cv::FileStorage file(path, cv::FileStorage::READ);

  file["trainDatacv::Size"] >> trainDataSize_;

  cv::FileNode edges_node = file["edges"];
  cv::FileNodeIterator it = edges_node.begin(), it_end = edges_node.end();

  bool is_first_vertex = true;
  int idx = 0;
  for (; it != it_end; ++it, idx++)
    {
      int source_idx = (int) (*it)["sourceIdx"];
      int target_idx = (int) (*it)["targetIdx"];
      double weight = (double) (*it)["weight"];

      verticies_count_map[source_idx] = true;
      verticies_count_map[target_idx] = true;

      Mat covariance, cov_inverted, mean, d, u;
      (*it)["cov"] >> covariance;
      (*it)["covInv"] >> cov_inverted;
      (*it)["sVec"] >> mean;
      (*it)["dVec"] >> d;
      (*it)["uVec"] >> u;

      TreeEdge e;
      bool inserted;
      tie(e, inserted) = add_edge(source_idx, target_idx, directedSpanningTree_);
      weightmap_directed_tree[e] = weight;

      if (is_first_vertex)
        {
          treeRoot_ = source(e, directedSpanningTree_);
          is_first_vertex = false;
        }

      E edge(source_idx, target_idx);
      tree_edges.push_back(edge);
      tree_weight_vector[edge] = weight;
      tree_covariances_vector[edge] = covariance;
      tree_covariances_vector_inverted[edge] = cov_inverted;
      tree_mean_mat_vector[edge] = mean;
      tree_mean_point_vector[edge] =
          cv::Point2d(mean.at<double>(0, 0), mean.at<double>(1, 0));
      tree_svd_d_vector[edge] = d;
      tree_svd_u_vector[edge] = u;
    }

  file.release();

  treeSize_ = verticies_count_map.size();

  // find leaves
  FindLeafVisitor find_leaves_visitor;
  depth_first_search(directedSpanningTree_, visitor(find_leaves_visitor));
}



double faceAnalysis::GlobalTree::CalcMahalanobis(const double li_x,
                                                 const double li_y,
                                                 const double lj_x,
                                                 const double lj_y) const {

  double diff_x = li_x - lj_x - current_edge_mean_.at<double>(0,0);
  double diff_y = li_y - lj_y - current_edge_mean_.at<double>(1,0);

  return sqrt((diff_x * current_edge_cov_inv_.at<double>(0,0)
               + diff_y * current_edge_cov_inv_.at<double>(1,0)) * diff_x
              + (diff_x * current_edge_cov_inv_.at<double>(0,1)
                 +diff_y * current_edge_cov_inv_.at<double>(1,1)) * diff_y);

}

void faceAnalysis::GlobalTree::UpdateEdge(const cv::Point2d better_li,
                                          const cv::Point2d better_lj) {
  Mat better_li_Mat(better_li);
  Mat better_lj_Mat(better_lj);
  Mat diff = better_li_Mat - better_lj_Mat;
  Mat new_mean = ((trainDataSize_) / (double) (trainDataSize_ + 1))
      * current_edge_mean_
      + ((1 / (double) (trainDataSize_ + 1)) * diff);

  tree_mean_mat_vector[current_edge_] = new_mean.clone();
  tree_mean_point_vector[current_edge_] =
      cv::Point2d(new_mean.at<double>(0, 0), new_mean.at<double>(1, 0));

  trainDataSize_++;
}


std::vector<TreeVertex> faceAnalysis::GlobalTree::childs(
    const TreeVertex treeVertex) const {
  std::vector<TreeVertex> childs;

  Tree::out_edge_iterator out_begin, out_end;
  for (boost::tie(out_begin, out_end) =
       out_edges(treeVertex,directedSpanningTree_); out_begin != out_end;
       ++out_begin)
    {
      childs.push_back(target(*out_begin, directedSpanningTree_));
    }

  return childs;
}

cv::Point2d faceAnalysis::GlobalTree::mean_point(const E edge) const
{
  return tree_mean_point_vector[edge];
}

