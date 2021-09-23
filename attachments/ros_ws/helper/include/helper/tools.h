#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <geometry_msgs/Point.h>
#include <helper/Object.h>
#include<fstream>

using namespace std;
using namespace Eigen;
using namespace geometry_msgs;
using namespace helper;
static bool init_matrix = false;
//static MatrixXf invRt, invTt;
class Tools{

public:

	Tools();
	~Tools();


	MatrixXf getImage2DBoundingBox(
		const Point & point,
		const float width,
		const float height);
	MatrixXf getImage2DBoundingBox(
		const Object o);

	// Transformation functions
	MatrixXf transformVeloToCam(const MatrixXf & velo_points);
	MatrixXf transformCamToRectCam(const MatrixXf & cam_points);
	MatrixXf transformRectCamToImage(const MatrixXf & rect_cam_points);
	MatrixXf transformVeloToImage(const MatrixXf & velo_points);
    MatrixXf pointcloud2_to_image(const MatrixXf & pointclound);

	int getClusterKernel(const int semantic);
//    void resetMatrix();
    void OutputMatrx(const MatrixXf & oMatrix, const char* fileName);
    void initMatrix(void);


	// Semantic helpers
	std::vector<std::string> SEMANTIC_NAMES;
	std::map<int, int> SEMANTIC_COLOR_TO_CLASS;
	MatrixXi SEMANTIC_CLASS_TO_COLOR;
	VectorXi SEMANTIC_KERNEL_SIZE;
	
private:

	// Transformation
	MatrixXf TRANS_VELO_TO_CAM;
	MatrixXf TRANS_CAM_TO_RECTCAM;
	MatrixXf TRANS_RECTCAM_TO_IMAGE;

    MatrixXf Rot_MAT;
    MatrixXf T_MAT;
    MatrixXf invT;
    MatrixXf DistCoeff;

};