#ifndef GLOBALTREE_H
#define	GLOBALTREE_H

#include <log4cxx/logger.h>
#include <opencv2/core/core.hpp>

#include "model/Types.h"

namespace faceAnalysis {
  class GlobalTree {
    static log4cxx::LoggerPtr logger;
  public:
    GlobalTree() {}
    explicit GlobalTree(const std::vector<int> landmarks_to_model);
    ~GlobalTree() {}

    double CalcMahalanobis(const double li_x, const double li_y,
                           const double lj_x, const double lj_y) const;

    void Load(const std::string path);
    void PrintOutput(const std::string path);
    void Save(const std::string path);
    void Train(const Dataset data);
    void UpdateEdge(const cv::Point2d better_li, const cv::Point2d better_lj);

    void set_landmarks_to_model(const std::vector<int> &landmarks_to_model)
    {
      landmarks_to_model_ = landmarks_to_model;
    }

    std::vector<TreeVertex> childs(const TreeVertex treeVertex) const;
    TreeVertex root() { return treeRoot_; }
    cv::Point2d mean_point(const E edge) const;
    std::map<int, cv::Point2d> all_means(const int rootID,
                                         const cv::Point2d root) const;


    void set_current_edge(const E &current_edge);

  private:
    std::vector<int> landmarks_to_model_;
    Tree directedSpanningTree_;
    TreeVertex treeRoot_;
    int treeSize_;
    int trainDataSize_;

    E current_edge_;
    cv::Mat current_edge_mean_;
    cv::Mat current_edge_cov_inv_;
  };

  typedef boost::shared_ptr<GlobalTree> GlobalTreePtr;
}
#endif	/* GLOBALTREE_H */

