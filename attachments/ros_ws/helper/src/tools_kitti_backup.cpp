#include <helper/tools.h>

Tools::Tools(){

	// Fill transformation matrices
	//TRANS_VELO_TO_CAM = MatrixXf::Zero(3, 4);
	printf("init TRANS_VELO_TO_CAM\n");
	TRANS_VELO_TO_CAM = MatrixXf::Zero(4, 4);
	TRANS_VELO_TO_CAM <<
            //1.4566160366765279e-02, 7.1136924172779004e-02,
            //9.9736019821898136e-01, 1.7643702449277043e-03,
            //-9.9951390620364911e-01, 2.8533054261184521e-02,
            //1.2562488608982791e-02, -3.5374432802200317e-02,
            //-2.7564075854129455e-02, -9.9705837483758408e-01,
            //7.1517961999369051e-02, -2.3568186163902283e-01;
 		7.533745000000e-03, -9.999714000000e-01, -6.166020000000e-04,
 		-4.069766000000e-03,  1.480249000000e-02,  7.280733000000e-04,
 		-9.998902000000e-01, -7.631618000000e-02,  9.998621000000e-01,
 		7.523790000000e-03,  1.480755000000e-02, -2.717806000000e-01,
 		0, 0 ,0 ,0;

    printf("init TRANS_CAM_TO_RECTCAM\n");
	TRANS_CAM_TO_RECTCAM = MatrixXf::Zero(4, 4);
	TRANS_CAM_TO_RECTCAM <<
	//1.0,0.0,0.0,0.0,
	//0.0,1.0,0.0,0.0,
	//0.0,0.0,1.0,0.0,
	//0.0,0.0,0.0,1.0;
 			9.999239000000e-01, 9.837760000000e-03, -7.445048000000e-03, 0,
 			-9.869795000000e-03, 9.999421000000e-01, -4.278459000000e-03, 0,
  			7.402527000000e-03, 4.351614000000e-03,  9.999631000000e-01, 0,
 			0, 0, 0, 1;

    printf("init TRANS_RECTCAM_TO_IMAGE\n");
	//TRANS_RECTCAM_TO_IMAGE = MatrixXf::Zero(3, 3);
	TRANS_RECTCAM_TO_IMAGE = MatrixXf::Zero(3, 4);
	TRANS_RECTCAM_TO_IMAGE <<
            //714.743408, 0.000000, 516.678831,
            //0.000000, 711.070190, 518.822461,
            //0.000000, 0.000000, 1.000000;
//                           714.743408, 0.000000, 516.678831, 0.000000,
//                           0.000000, 711.070190, 518.822461, 0.000000,
//                           0.000000, 0.000000, 1.000000, 0.000000;
		7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02,
 		4.485728000000e+01, 0.000000000000e+00, 7.215377000000e+02,
 		1.728540000000e+02, 2.163791000000e-01, 0.000000000000e+00,
 		0.000000000000e+00, 1.000000000000e+00, 2.745884000000e-03;

    printf("init Rot_MAT\n");
    Rot_MAT = MatrixXf::Zero(3, 3);
    Rot_MAT <<
             -9.8836088830724711e-01, 1.1326651711791161e-02,
             -1.5170517929691568e-01,
             1.5212676276466525e-01, 7.0638167047866629e-02,
             -9.8583350389751967e-01,
             -4.5001694725983754e-04, -9.9743769545627137e-01,
             -7.1539088389702687e-02;
    //Huksy
          //1.4566160366765279e-02, 7.1136924172779004e-02,9.9736019821898136e-01,
          //-9.9951390620364911e-01, 2.8533054261184521e-02,1.2562488608982791e-02,
          //-2.7564075854129455e-02, -9.9705837483758408e-01,7.1517961999369051e-02;

    printf("init T_MAT\n");
    T_MAT = MatrixXf::Zero(3, 1);
    T_MAT <<
          1.2438613176345825e-01, -8.5932992398738861e-02, -1.9025887548923492e-01;
    //Husky
            // 1.7643702449277043e-03,-3.5374432802200317e-02,-2.3568186163902283e-01;

    DistCoeff = MatrixXf::Zero(1, 5);
    DistCoeff << -1.4080599999999999e-01, 9.2914999999999998e-02,
    1.8760000000000001e-03, 1.1969999999999999e-03, 0.;
    //initMatrix();


	SEMANTIC_NAMES = std::vector<std::string>{
		// Static objects
		"Road", "Sidewalk", "Building", "Wall", "Fence", "Pole",
		"Traffic light", "Traffic sign", "Vegetation", "Terrain", "Sky",
		// Dynamic objects
		"Pedestrian", "Rider", "Car", "Truck", "Bus", "Train", "Motocycle", "Bicycle"
	};

	SEMANTIC_COLOR_TO_CLASS = std::map<int, int>{
		// Static objects
		{320, 0}, {511, 1}, {210, 2}, {260, 3}, {496, 4}, {459, 5},
		{450, 6}, {440, 7}, {284, 8}, {555, 9}, {380, 10},
		// Dynamic objects
		{300, 11}, {255, 12}, {142, 13}, {70, 14},{160, 15}, {180, 16}, {230, 17}, {162, 18}
	};

	SEMANTIC_CLASS_TO_COLOR = MatrixXi::Zero(19, 3);
	SEMANTIC_CLASS_TO_COLOR <<
	// Static objects
		128,  64, 128, // Road
		244,  35, 232, // Sidewalk
		 70,  70,  70, // Building
		102, 102, 156, // Wall
		190, 153, 153, // Fence
		153, 153, 153, // Pole
		250, 170,  30, // Traffic light
		220, 220,   0, // Traffic sign
		107, 142,  35, // Vegetation
		152, 251, 152, // Terrain
		 70, 130, 180, // Sky

	// Dynamic objects
		220,  20,  60, // Pedestrian
		255,   0,   0, // Rider
		  0,   0, 142, // Car
		  0,   0,  70, // Truck
		  0,  60, 100, // Bus
		  0,  80, 100, // Train
		  0,   0, 230, // Motocycle
		119,  11,  32;  // Bicycle

	SEMANTIC_KERNEL_SIZE = VectorXi::Zero(8);	
	SEMANTIC_KERNEL_SIZE <<
		1, // Pedestrian
		2, // Rider
		1, // Car
		4, // Truck
		5, // Bus
		5, // Train
		2, // Motocycle
		2; // Bicycle
}

Tools::~Tools(){

}

int Tools::getClusterKernel(const int semantic){

	if(semantic > 10)
		return SEMANTIC_KERNEL_SIZE(semantic - 11);
	else
		return -1;
}

MatrixXf Tools::getImage2DBoundingBox(
	const Point & point,
	const float width,
	const float height){

	MatrixXf velo_points = MatrixXf::Zero(4,2);
	velo_points(0,0) = point.x;
	velo_points(1,0) = point.y + width;
	velo_points(2,0) = point.z + height;
	velo_points(3,0) = 1;
	velo_points(0,1) = point.x;
	velo_points(1,1) = point.y - width;
	velo_points(2,1) = point.z;
	velo_points(3,1) = 1;
	MatrixXf image_points = transformVeloToImage(velo_points);
	return image_points;
}

MatrixXf Tools::getImage2DBoundingBox(
	const Object o){

	// Rotate top view box with velo orientation
	float rad_ori = o.orientation / 180 * M_PI;

	float half_length = o.length / 2;
	float half_width = o.width / 2;
	float cos_l = half_length * cos(rad_ori);
	float sin_w = half_width * sin(rad_ori);
	float sin_l = half_length * sin(rad_ori);
	float cos_w = half_width * cos(rad_ori);

	MatrixXf velo_points = MatrixXf::Zero(4,8);
	velo_points(0,0) = o.velo_pose.point.x + cos_l + sin_w;
	velo_points(1,0) = o.velo_pose.point.y + sin_l - cos_w;
	velo_points(2,0) = o.velo_pose.point.z + o.height;
	velo_points(3,0) = 1;

	velo_points(0,1) = o.velo_pose.point.x + cos_l - sin_w;
	velo_points(1,1) = o.velo_pose.point.y + sin_l + cos_w;
	velo_points(2,1) = o.velo_pose.point.z + o.height;
	velo_points(3,1) = 1;

	velo_points(0,2) = o.velo_pose.point.x - cos_l + sin_w;
	velo_points(1,2) = o.velo_pose.point.y - sin_l - cos_w;
	velo_points(2,2) = o.velo_pose.point.z + o.height;
	velo_points(3,2) = 1;

	velo_points(0,3) = o.velo_pose.point.x - cos_l - sin_w;
	velo_points(1,3) = o.velo_pose.point.y - sin_l + cos_w;
	velo_points(2,3) = o.velo_pose.point.z + o.height;
	velo_points(3,3) = 1;

	velo_points(0,4) = o.velo_pose.point.x + cos_l + sin_w;
	velo_points(1,4) = o.velo_pose.point.y + sin_l - cos_w;
	velo_points(2,4) = o.velo_pose.point.z;
	velo_points(3,4) = 1;

	velo_points(0,5) = o.velo_pose.point.x + cos_l - sin_w;
	velo_points(1,5) = o.velo_pose.point.y + sin_l + cos_w;
	velo_points(2,5) = o.velo_pose.point.z;
	velo_points(3,5) = 1;

	velo_points(0,6) = o.velo_pose.point.x - cos_l + sin_w;
	velo_points(1,6) = o.velo_pose.point.y - sin_l - cos_w;
	velo_points(2,6) = o.velo_pose.point.z;
	velo_points(3,6) = 1;

	velo_points(0,7) = o.velo_pose.point.x - cos_l - sin_w;
	velo_points(1,7) = o.velo_pose.point.y - sin_l + cos_w;
	velo_points(2,7) = o.velo_pose.point.z;
	velo_points(3,7) = 1;

	//MatrixXf image_points = pointcloud2_to_image(velo_points);
	MatrixXf image_points = transformVeloToImage(velo_points);

	float min_x = image_points(0,0);
	float max_x = image_points(0,0);
	float min_y = image_points(1,0);
	float max_y = image_points(1,0);
	for(int i = 1; i < 8; i++){
		min_x = (min_x < image_points(0,i)) ? min_x : image_points(0,i);
		max_x = (max_x > image_points(0,i)) ? max_x : image_points(0,i);
		min_y = (min_y < image_points(1,i)) ? min_y : image_points(1,i);
		max_y = (max_y > image_points(1,i)) ? max_y : image_points(1,i);
	}

	// Check bounding
	if(min_x < 0)
		min_x = 0.0;
	if(max_x > 1237)
		max_x = 1237.0;
	if(min_y < 0)
		min_y = 0.0;
	if(max_y > 370)
		max_y = 370.0;

	MatrixXf box = MatrixXf::Zero(2,2);
	box(0,0) = min_x;
	box(1,0) = min_y;
	box(0,1) = max_x;
	box(1,1) = max_y;

	return box;
}

MatrixXf Tools::transformVeloToCam(const MatrixXf & velo_points){

	return TRANS_VELO_TO_CAM * velo_points;
}

MatrixXf Tools::transformCamToRectCam(const MatrixXf & cam_points){

	return TRANS_CAM_TO_RECTCAM * cam_points;
}

MatrixXf Tools::transformRectCamToImage(const MatrixXf & rect_cam_points){

	MatrixXf image_points = TRANS_RECTCAM_TO_IMAGE * rect_cam_points;
	MatrixXf uv = MatrixXf::Zero(3,rect_cam_points.cols());
	uv.row(0) = image_points.row(0).array()/image_points.row(2).array();
	uv.row(1) = image_points.row(1).array()/image_points.row(2).array();
	uv.row(2) = image_points.row(2);
	return uv;
}

MatrixXf Tools::transformVeloToImage(const MatrixXf & velo_points){

	return transformRectCamToImage(TRANS_CAM_TO_RECTCAM * TRANS_VELO_TO_CAM * velo_points);
}

void Tools::initMatrix(void)
{
    invT = MatrixXf::Zero(1,3);
    invT = -1.0 * Rot_MAT.transpose() * T_MAT;
    TRANS_VELO_TO_CAM << Rot_MAT.transpose(), invT;
    std::cout<<TRANS_VELO_TO_CAM<<std::endl;
}

MatrixXf Tools::pointcloud2_to_image(const MatrixXf & pointclound)
{
    MatrixXf velo2img =  TRANS_VELO_TO_CAM * pointclound;
    //(Rot_MAT * pointclound).array() + invT.array(); //3*n + MatrixXf::Ones(3, pointclound.cols()).cwiseProduct(invT)
    MatrixXf img_points = MatrixXf::Zero( 2, pointclound.cols());
    img_points.row(0) = velo2img.row(0).array()/velo2img.row(2).array();
    img_points.row(1) = velo2img.row(1).array()/velo2img.row(2).array();
//    OutputMatrx(img_points.row(0), "img_points_row1.txt");
//    OutputMatrx(img_points.row(1), "img_points_row2.txt");
    MatrixXf radius =  MatrixXf::Zero(1, pointclound.cols());
    MatrixXf dist =  MatrixXf::Zero( 1, pointclound.cols());
    radius = img_points.colwise().squaredNorm();
//
    dist = MatrixXf::Ones(1, pointclound.cols()).array() + DistCoeff(0,0) * radius.array() +
            DistCoeff(0,1)*radius.array().square() + DistCoeff(0,2)*radius.array().cube();
    MatrixXf uv = MatrixXf::Ones(3, pointclound.cols());

    MatrixXf row0_term1 =  MatrixXf::Zero( 1, pointclound.cols());
    row0_term1 = img_points.row(0).cwiseProduct(dist);
    MatrixXf row0_term2 =  MatrixXf::Zero( 1, pointclound.cols());
    row0_term2 = 2 * DistCoeff(0,4) * img_points.row(0).array() * img_points.row(1).array();
    MatrixXf row0_term3 =  MatrixXf::Zero( 1, pointclound.cols());
    row0_term3 = DistCoeff(0,3) * (radius.array() + 2 * img_points.row(0).array().square());
    uv.row(0) = row0_term1 + row0_term2 + row0_term3;
//    uv.row(0) = img_points.row(0).cwiseProduct(dist) + 2 * DistCoeff(0,2) * img_points.row(0) * img_points.row(1)
//            + DistCoeff(0,3) * (radius + 2 * img_points.row(0) * img_points.row(1));

    MatrixXf row1_term1 =  MatrixXf::Zero( 1, pointclound.cols());
    row1_term1 = img_points.row(1).cwiseProduct(dist);
    MatrixXf row1_term2 =  MatrixXf::Zero( 1, pointclound.cols());
    row1_term2 = DistCoeff(0,3) * (radius.array() + 2 * img_points.row(1).array().square()); //p1(r^2+2y^2)
    MatrixXf row1_term3 =  MatrixXf::Zero( 1, pointclound.cols());
    row1_term3 = 2 * DistCoeff(0,4) * img_points.row(0).array() * img_points.row(1).array(); //2p_2*x*y
    uv.row(1) = row1_term1 + row1_term2 + row1_term3;
//    uv.row(1) = img_points.row(1).cwiseProduct(dist) + 2 * DistCoeff(0,2) * (radius + 2 * img_points.row(0) * img_points.row(1))
//                + DistCoeff(0,3) * img_points.row(0) * img_points.row(1);

//    uv.col(0) = uv.col(0) * TRANS_RECTCAM_TO_IMAGE(0,0) + TRANS_RECTCAM_TO_IMAGE(0,2);
//    uv.col(1) = uv.col(1) * TRANS_RECTCAM_TO_IMAGE(1,1) + TRANS_RECTCAM_TO_IMAGE(1,2);
    uv = TRANS_RECTCAM_TO_IMAGE * uv;
    uv.row(2) = velo2img.row(2);
//    OutputMatrx(uv.row(0), "row1.txt");
//    OutputMatrx(uv.row(1), "row2.txt");
//    OutputMatrx(uv, "uv.txt");
    return uv;
}

void Tools::OutputMatrx(const MatrixXf & oMatrix, const char* fileName)
{
    std::ofstream output_file(fileName, std::ios::out | std::ios::ate);

    if (output_file.is_open())
    {
        std::cout << "Open file successfully." << std::endl;
        output_file << oMatrix;
        output_file.close();
        std::cout << "Writing down!" << std::endl;
    } else  std::cout << "Unable to open output file" << std::endl;
    std::cout << "*********************************" << std::endl;

}