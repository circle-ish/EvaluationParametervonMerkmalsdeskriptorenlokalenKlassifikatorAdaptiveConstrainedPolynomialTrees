#ifndef CPT_H
#define	CPT_H

#include <log4cxx/logger.h>
#include <opencv2/core/core.hpp>

#include "model/GlobalConstants.h"
#include "model/global/GlobalTree.h"
#include "model/local/LocalPolynoms.h"


namespace faceAnalysis {
  class CPT {
    static log4cxx::LoggerPtr logger;

    //static const int FEATURE_SIZE = 283; //static const int FEATURE_SIZE = 81;
    //static const int FEATURE_SIZE = 162; // size of the feature to train the local classificator
    static const int GAUSS_RADIUS_UPDATE = 2; // gauss to use for update polynom
    static const int SEARCHWINDOW_RADIUS = 20; // windowradius around each vertex to search for a minimum
    static const int SURROUNDING_MIDPOINT_DETECTION_RADIUS = 15; // windowradius around the root to search for a minimum
    static const int SURROUNDING_TO_CHECK = 20;
  public:
    explicit CPT(const boost::property_tree::ptree configuration,
                 std::vector<int> landmarksToUse = std::vector<int>(),
                 std::string output_folder_path = "");

    ~CPT() {}

    void ClearPoints();

    void CollectGlobalMeans(const TreeVertex current_vertex);

    void DetectFeatures(const bool reinit,
                        const cv::Point2f rootPoint = cv::Point2f(0,0));

    void SetImage(const cv::Mat &image, const cv::Rect& detection);


    void FreeBc();
    void FreeBcPosition();
    void FreePolynoms();

    void GenerateB_c(const int image_cols,const int image_rows);

    void Load(const std::string path_string);

    void UpdateLocal(const cv::Point2i position, const int polynom_id);
    void UpdateGlobal(const cv::Point2i better_li, const cv::Point2i better_lj);


    void SetElementBc(const int idx, const int jdx, const int kdx,
                      const double value);
    double GetElementBc(int idx, int jdx, int kdx) const;

    void SetElementBcPosition(const int idx, const int jdx, const int kdx,
                              const int point_x, const int point_y);
    std::pair<int, int> GetElementBcPosition(int idx, int jdx, int kdx) const;


    std::map<int, std::pair<int, int> > unscaled_collected_points() ;

    int root_ID() { return ((int) globalModel_->root()); }

  private:
    double AnswerChild(
        const int parent_pos_x, const int parent_pos_y,
        const TreeVertex current_vertex, const E current_edge);

    void CollectPositions(bool recursion_first_time, int parent_pos_x, int parent_pos_y,
                          TreeVertex current_vertex, E current_edge);

    double *b_c_;
    int b_c_size_x;
    int b_c_size_y_;
    int b_c_size_z_;

    int *b_c_position_;

    cv::Rect current_face_detection_;
    cv::Mat current_image_;
    double current_scale_factor_;

    cv::Mat scaled_face_;
    cv::Mat scaled_whole_image_;


    std::map<int, cv::Point2d> all_face_points_means_;

    std::map<int, std::pair<int, int> > collected_face_points_;

    std::map<int, double> collected_face_points_values_;

    GlobalTreePtr globalModel_;
    LocalPolynomsPtr localPolynoms_;

    cv::Mat gui_;

    std::string output_folder_path_;

    bool imageSet;


    double weight_local_;
    double weight_global_;

    std::string global_model_file_name_;

    bool first_detection_;
  };

  typedef boost::shared_ptr<CPT> CPTPtr;
}

#endif	/* CPT_H */
