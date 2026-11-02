#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <cmath>
namespace fs = std::filesystem;

int main()
{
    if (!fs::exists("./out"))
    {
        fs::create_directory("./out");
    }
    for (auto &entry : fs::directory_iterator("./input"))
    {
        cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
        if (img.empty())
        {
            std::cout << "错误:未输入图片" << std::endl;
            continue;
        }
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        cv::Mat blur_img;
        cv::GaussianBlur(gray, blur_img, cv::Size(5,5), 0);
        cv::Mat dx;
        cv::Mat dy;
        cv::Sobel (blur_img, dx, CV_32F, 1, 0, 3);
        cv::Sobel (blur_img, dy, CV_32F, 0, 1, 3);
        cv::Mat mag;
        cv::magnitude (dx, dy, mag);
        cv::Mat mag_8U;
        cv::normalize(mag, mag_8U, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::Mat waste;
        double threshold = cv::threshold(mag_8U, waste, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        double low =  threshold * 0.5;
        cv::Mat edge;
        cv::Canny(blur_img, edge, low, threshold);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(edge, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        int rectcount = 0;
        std::string resultLine; 
        for(size_t i = 0; i < contours.size(); i++ )
        {
            double length = cv::arcLength(contours[i], true);
            std::vector<cv::Point> approx;
            cv::approxPolyDP(contours[i], approx, 0.02 * length, true);
            if (approx.size() == 4 && isContourConvex(approx))
            {
                auto p0 = approx[0];
                auto p1 = approx[1];
                auto p2 = approx[2];
                auto p3 = approx[3];
                auto M1 = (p0 + p2) / 2;//对角线中点1
                auto M2 = (p1 + p3) / 2;//对角线中点2
                double dx1 = p2.x - p0.x;
                double dy1 = p2.y - p0.y;
                double d1 = std::sqrt(dx1*dx1 + dy1*dy1);//对角线1
                double dx2 = p3.x - p1.x;
                double dy2 = p3.y - p1.y;
                double d2 = std::sqrt(dx2*dx2 + dy2*dy2);//对角线2
                double dmy = M1.y - M2.y;//计算两个对角线中点之间的距离
                double dmx = M1.x - M2.x;
                double point_difference = sqrt(dmy*dmy + dmx*dmx);
                double diagonal_diffenrence = std::abs (d1 - d2) / std::max(d1, d2);
                if (point_difference < 2 && diagonal_diffenrence < 0.05)
                {
                    cv::polylines(img, approx, true, cv::Scalar(0, 255, 0), 2);
                    rectcount++;
                    std:: string part = "rect" + std::to_string(rectcount);
                    for (int k = 0; k < 4; k++)
                    {
                        int x_out = approx[k].x;
                        int y_out = img.rows - 1 - approx[k].y;
                        part += "(" + std::to_string(x_out) + "," + std::to_string(y_out) + ")";
                        if (k < 3)
                        part += ";";
                    }
                    if (!resultLine.empty())
                    {
                        resultLine += "|";
                    }
                    resultLine += part;
                }
            }
        }
        std::string name = entry.path().filename();
        if (rectcount == 0)
        std::cout << name << ": null" << std::endl;
        else
        std::cout << name << ": " << resultLine << std::endl;
        fs::path out_path = "./out" / entry.path().filename();
        cv::imwrite(out_path.string(),img);
    }
}