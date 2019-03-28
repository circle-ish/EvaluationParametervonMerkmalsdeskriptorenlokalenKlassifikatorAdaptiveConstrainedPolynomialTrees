#ifndef LOCALPOLYNOMS_H
#define	LOCALPOLYNOMS_H

#include <fstream>
#include <log4cxx/logger.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "model/local/Polynom.h"
#include "model/local/PatchFeature.h"
#include "model/local/PatchFeatureUtils.h"
#include "model/local/ModifiedPCA.h"
#include "model/local/Gauss.h"
#include "model/dataHandlers/DatasetHandler.h"
#include "model/Types.h"

namespace faceAnalysis {
  class LocalPolynoms {
    static log4cxx::LoggerPtr logger;
    static const int GAUSS_RADIUS = 2;

  public:
    explicit LocalPolynoms(
            const boost::property_tree::ptree configuration,
            const std::vector<int> landmarks_to_model = std::vector<int>()
            );
    ~LocalPolynoms() {}

    void Save();
    void Load(std::string path);

    void ClearAllPolynoms();
    void FreeAllPolynoms();
    void FreeFeatureMap();

    void Train(const Dataset data);
    void Update(int polynom_id, int x, int y, double value, double weight);
    void SetCurrentImage(const cv::Mat& image);
    cv::Point2i GetMaxValuePosition(int polynom_id);
    cv::Point2i GetMaxValuePositionBySum(int polynom_id);

    void InitFeatureMap(const int image_rows, const int image_cols);
    void GenerateFeature(
            const int row_id,
            const int col_id
            );
    /** generates a feature vector for every pixel in current_image_*/
    void ExtractFeatures(std::vector<std::vector<double> > &features);
    void GenerateFeatureMap(
            std::vector<std::vector<double> > &features,
            const int polyDegree,
            const int landmark_index,
            std::string eigenPath
            );
    void SetElementFeatureMap(const int row_id, const int col_id, const int deep_id,
                              const double value);
    double GetElementFeatureMap(
            int row_id,
            int col_id,
            int depth_id
            ) const;

    void initAnswer(int polynom_id);
    double GenerateAnswerForPolynom(int polynom_id, int row_id, int col_id);
    void GenerateSumAnswerForPolynom(int polynom_id);
    double getAnswer(int polynom_id, int row_id, int col_id);
    bool IsAnswerMapInitialized(int polynom_id);
    void DrawAnswerMap(int polynom_id);

    int Size();

  private:
    void generateLocalTrainData(
            const Dataset &data,
            int point_counter,
            int size_landmarks_to_model,
            int landmark_index
            );
    /** builds feature vector for the given methods (useFeature) and
     * image (src)
     * return a vector of doubles (features) */
    void buildFeatureVectors(
            const cv::Mat &src,
            std::vector<std::vector<double> > &features,
            const int useFeature
            );


    //conf
    boost::property_tree::ptree configuration_;
    std::string relative_path_to_save_location_;
    std::string path_to_save_location_;

    //polynom
    std::map<int, Polynom> allPolynoms_;
    std::map<int, Polynom> polynoms_;
    bool sort_polynom_;
    ClassifierType classifier_type_;
    cv::Mat c_;
    cv::Mat y_;

    //patchfeature
    PatchFeature patch_feature_;
    Gauss gauss_;
    /** every dimension a vectors;
     * access with boundary checks with .at(); without check with [] */
    Mat3D_t feature_map_;
    DatasetHandler dataHandler_;

    //params
    int random_negative_examples_total_;
    std::vector<int> landmarks_to_model_;
    cv::Mat current_image_;

    /** times for eval */
    double timeHOG, timeCoHOG, timeMrLBP, timeGrad, timeLBP;
  };

  typedef boost::shared_ptr<LocalPolynoms> LocalPolynomsPtr;

}

#endif	/* LOCALPOLYNOMS_H */

