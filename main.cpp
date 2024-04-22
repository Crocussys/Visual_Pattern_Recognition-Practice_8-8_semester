// #include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;

struct card
{
    const char* name;
    Mat image;
    vector<KeyPoint> keypoints;
    Mat descriptors;
};

Mat frame;
vector<card> cards;
int blure, thresh;

Ptr<SIFT> siftPtr = SIFT::create();
Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);

static void on_trackbar(int, void*){
    Mat mask, temp;

    cvtColor(frame, temp, COLOR_BGR2GRAY);
    if (blure > 0) GaussianBlur(temp, temp, Size(0, 0), blure);
    threshold(temp, mask, thresh, 255, THRESH_BINARY);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    vector<Point> approx;

    findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for(size_t i = 0; i < contours.size(); i++){
        approxPolyDP(contours[i], approx, arcLength(contours[i], true) * 0.02, true);
        if (fabs(contourArea(contours[i])) < 1000 || !isContourConvex(approx))
            continue;
        if (approx.size() == 4 && cv::isContourConvex(approx)){
            drawContours(frame, contours, (int)i, Scalar(0, 255, 0), 3, LINE_8, hierarchy, 0);
            Mat card_mask = Mat::zeros(mask.size(), mask.type());
            fillPoly(card_mask, contours[i], 255);
            Mat descriptors;
            vector<KeyPoint> keypoints;
            siftPtr->detectAndCompute(temp, card_mask, keypoints, descriptors);
            for (int j = 0; j < cards.size(); j++){
                std::vector<DMatch> matches;
                matcher->match(descriptors, cards[j].descriptors, matches);
                double dist = 0;
                for (size_t j = 0; j < matches.size(); ++j) {
                    dist += matches[j].distance;
                }
                dist /= matches.size();
                if (dist < DBL_MAX) {
                    Moments m = moments(approx, true);
                    Point p(m.m10/m.m00 - 150, m.m01/m.m00);
                    putText(frame, cards[j].name, p, 0, 2, Scalar(0, 255, 0), 10);
                    break;
                }
            }
        }
    }
    imshow("Frame", frame);
}

void add_card(const char path[], const char name[]){
    card new_card;
    new_card.image = imread(path);
    new_card.name = name;
    Mat mask;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    if (blure > 0) GaussianBlur(new_card.image, new_card.image, Size(0, 0), blure);
    cvtColor(new_card.image, new_card.image, COLOR_BGR2GRAY);
    threshold(new_card.image, mask, thresh, 255, THRESH_BINARY);
    findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    Mat card_mask = Mat::zeros(mask.size(), mask.type());
    fillPoly(card_mask, contours[0], 255);
    siftPtr->detectAndCompute(new_card.image, card_mask, new_card.keypoints, new_card.descriptors);
    cards.push_back(new_card);
    // Mat temp;
    // drawKeypoints(new_card.image, new_card.keypoints, temp);
}

int main()
{
    blure = 7;
    thresh = 156;
    namedWindow("Frame", WINDOW_NORMAL);
    createTrackbar("blure", "Frame", &blure, 20, on_trackbar);
    createTrackbar("thresh", "Frame", &thresh, 1000, on_trackbar);

    add_card("../Visual_Pattern_Recognition-Practice_8-8_semester/card1.jpg", "9 of Clubs");
    add_card("../Visual_Pattern_Recognition-Practice_8-8_semester/card2.jpg", "King of Diamonds");
    add_card("../Visual_Pattern_Recognition-Practice_8-8_semester/card3.jpg", "Ace of Hearts");
    add_card("../Visual_Pattern_Recognition-Practice_8-8_semester/card4.jpg", "Queen of Spades");
    add_card("../Visual_Pattern_Recognition-Practice_8-8_semester/card5.jpg", "Jack of Hearts");
    add_card("../Visual_Pattern_Recognition-Practice_8-8_semester/card6.jpg", "6 of Spades");
    add_card("../Visual_Pattern_Recognition-Practice_8-8_semester/card7.jpg", "Ace of Diamonds");

    frame = imread("../Visual_Pattern_Recognition-Practice_8-8_semester/test.jpg");
    on_trackbar(blure, 0);
    waitKey(0);


    // VideoCapture cap("../Visual_Pattern_Recognition-Practice_8-8_semester/video.mp4");
    // if(!cap.isOpened()){
    //     cout << "Error" << endl;
    //     return -1;
    // }
    // VideoWriter out("../Visual_Pattern_Recognition-Practice_8-8_semester/output.mp4", cap.get(CAP_PROP_FOURCC), cap.get(CAP_PROP_FPS), Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));
    // bool start = false;
    // for (cap >> frame; !frame.empty(); cap >> frame){
    //     on_trackbar(blure, 0);
    //     out.write(frame);
    //     char c = (char) waitKey(1);
    //     if (c == 27) break;
    //     if (c == 32 || start){
    //         while(true){
    //             char c = (char) waitKey(1);
    //             if (c == 32) break;
    //         }
    //         start = false;
    //     }
    // }
    // cap.release();
    // out.release();
    destroyAllWindows();
    return 0;
}
