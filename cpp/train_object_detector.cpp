
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>

#include <dlib/svm_threaded.h>
#include <dlib/string.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include <dlib/data_io.h>
#include <dlib/cmd_line_parser.h>


#include <iostream>
#include <fstream>


using namespace std;
using namespace dlib;
using namespace cv;

int main(int argc, char** argv){
    try
    {
    command_line_parser parser;
    parser.parse(argc, argv);
    const unsigned long upsample_amount =1; 
    const unsigned long image_batch =1;
    typedef scan_fhog_pyramid<pyramid_down<120> > image_scanner_type_small;
    typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type_large;
    
    //load model
    ifstream fin("../SideEyNose.svm", ios::binary);
    if (!fin)
        {
            cout << "Can't find a trained object detector file SideEyNose.svm. " << endl;
            cout << "You need to train one using the -t option." << endl;
            cout << "\nTry the -h option for more information." << endl;
            return EXIT_FAILURE;
        }
    object_detector<image_scanner_type_large> detector;
    deserialize(detector, fin);

    ifstream fin_ear("../SideEar.svm", ios::binary);
    if (!fin_ear)
        {
            cout << "Can't find a trained object detector file SideEyNose.svm. " << endl;
            cout << "You need to train one using the -t option." << endl;
            cout << "\nTry the -h option for more information." << endl;
            return EXIT_FAILURE;
        }
    object_detector<image_scanner_type_large> ear_detector;
    deserialize(ear_detector, fin_ear);

    //use opencv to load image and transfer into datatype dlib can work with
    Mat image;
    array2d<bgr_pixel> cimg;
    image = imread(parser[0], CV_LOAD_IMAGE_COLOR); 
    assign_image(cimg, cv_image<bgr_pixel>(image));    
    //type 1 to use upsampled image
    if (parser[1] == "1"){
    for (unsigned long i = 0; i < upsample_amount; ++i){
        cout<<"Upsample image"<<endl;
        pyramid_up(cimg);
        }
    }


    //detect face
    image_window win;
    for (unsigned long i = 0; i < image_batch; ++i){
        const std::vector<rectangle> rects = detector(cimg);
        cout << "Number of Nose detections: "<< rects.size() << endl;

        const std::vector<rectangle> ear_rects = ear_detector(cimg);
        cout << "Number of Ear detections: "<< ear_rects.size() << endl;
        
        win.clear_overlay();
        win.set_image(cimg);
        win.add_overlay(rects, rgb_pixel(255,0,0));
        win.add_overlay(ear_rects, rgb_pixel(255,255,0));
        cout << "Hit enter to see the next image.";
        cin.get();
        }
    
    
    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
        cout << "\nTry the -h option for more information." << endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
