// =============================================================================
// CIFAR-10 Perceptron / MLP example
// =============================================================================

#if defined(__has_include)
#  if __has_include(<opencv2/opencv.hpp>) && __has_include(<opencv2/objdetect.hpp>)
#    include <opencv2/opencv.hpp>
#    include <opencv2/objdetect.hpp>
#    define HAVE_OPENCV 1
#  endif
#endif

#ifndef HAVE_OPENCV
// Minimal stubs for environments without OpenCV to allow compilation only.
#include <cstddef>
#include <vector>
#include <string>
#include <iostream>
// Provide common OpenCV type macros when OpenCV is not available
#define CV_8UC1 0
#define CV_32FC1 1

namespace cv {
    struct Size { int width, height; Size(int w=0,int h=0):width(w),height(h){} }; 
    enum { COLOR_BGR2GRAY = 0 };
    class Mat {
    public:
        void* data;
        int rows, cols, type;
        Mat(): data(nullptr), rows(0), cols(0), type(0) {}
        Mat(int r, int c, int t, void* d): data(d), rows(r), cols(c), type(t) {}
        void convertTo(Mat& dst, int) { dst = *this; }
        Mat clone() const { return *this; }
        Mat& operator/=(float) { return *this; }
    };
    inline void cvtColor(const Mat& src, Mat& dst, int) { dst = src; }
    inline void merge(const std::vector<Mat>&, Mat& dst) { dst = Mat(); }
    class HOGDescriptor {
    public:
        HOGDescriptor(Size, Size, Size, Size, int) {}
        void compute(const Mat&, std::vector<float>&, Size, Size) {}
    };
}
#else
// OpenCV available
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#endif

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include <windows.h>
#include <cmath>
#include <ctime>

static inline float Sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

// Read CIFAR binary
int ReadCifar(const std::string& filename,
    std::vector<cv::Mat>* images,
    std::vector<int>* labels,
    std::vector<int>* labels2, bool cifar100) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs.is_open()) {
        std::cout << "[ERROR] cannot open " << filename << std::endl;
        return 0;
    }
    unsigned char label, label2 = 0;
    unsigned char bufferR[1024], bufferG[1024], bufferB[1024];
    int cnt = 0;

    while (ifs.read((char*)&label, 1)) {
        if (cifar100) ifs.read((char*)&label2, 1);
        ifs.read((char*)bufferR, 1024);
        ifs.read((char*)bufferG, 1024);
        ifs.read((char*)bufferB, 1024);

        cv::Mat R(32, 32, CV_8UC1, bufferR);
        cv::Mat G(32, 32, CV_8UC1, bufferG);
        cv::Mat B(32, 32, CV_8UC1, bufferB);
        cv::Mat image;
        std::vector<cv::Mat> channels;
        channels.push_back(B);
        channels.push_back(G);
        channels.push_back(R);
        cv::merge(channels, image);

        labels->push_back((int)label);
        if (labels2) labels2->push_back((int)label2);
        images->push_back(image.clone());
    }
}

// Perceptron
class Perceptron {
public:
    int InputNum, OutputNum;
    float** Weight;
    float* BiasOut;
    float* Output;
    float LearningRate;

    Perceptron(int nIn, int nOut, float lr)
        : InputNum(nIn), OutputNum(nOut), LearningRate(lr) {
        Weight = new float* [OutputNum];
        BiasOut = new float[OutputNum];
        Output = new float[OutputNum];
        for (int i = 0; i < OutputNum; ++i) {
            Weight[i] = new float[InputNum];
            for (int k = 0; k < InputNum; ++k)
                Weight[i][k] = (float)rand() / RAND_MAX - 0.5f;
            BiasOut[i] = (float)rand() / RAND_MAX - 0.5f;
        }
    }
    ~Perceptron() {
        for (int i = 0; i < OutputNum; ++i) delete[] Weight[i];
        delete[] Weight; delete[] BiasOut; delete[] Output;
    }

    int Compute(float* Input) {
        for (int i = 0; i < OutputNum; ++i) {
            float sum = BiasOut[i];
            for (int k = 0; k < InputNum; ++k)
                sum += Weight[i][k] * Input[k];
            Output[i] = Sigmoid(sum);
        }
        float* max_pos = std::max_element(Output, Output + OutputNum);
        return (int)std::distance(Output, max_pos);
    }

    float Train(float* Input, int nClass) {
        Compute(Input);
        float error = 0.f;
        for (int i = 0; i < OutputNum; ++i) {
            float target = (i == nClass) ? 1.f : 0.f;
            float e = target - Output[i];
            float delta = e * Output[i] * (1.f - Output[i]);
            for (int k = 0; k < InputNum; ++k)
                Weight[i][k] += LearningRate * delta * Input[k];
            BiasOut[i] += LearningRate * delta;
            error += e * e;
        }
        return error;
    }
};

// 1-hidden-layer MLP
class Mlp {
public:
    int InputNum, OutputNum, HiddenNum;
    float** WeightOut, ** WeightHidden, * Output, * Hidden, * HiddenDelta;
    float* BiasOut, * BiasHidden;
    float LearningRate;

    Mlp(int nIn, int nHidden, int nOut, float lr)
        : InputNum(nIn), HiddenNum(nHidden), OutputNum(nOut), LearningRate(lr) {
        WeightOut = new float* [OutputNum];
        BiasOut = new float[OutputNum];
        for (int i = 0; i < OutputNum; ++i) {
            WeightOut[i] = new float[HiddenNum];
            for (int k = 0; k < HiddenNum; ++k)
                WeightOut[i][k] = (float)rand() / RAND_MAX - 0.5f;
            BiasOut[i] = (float)rand() / RAND_MAX - 0.5f;
        }
        WeightHidden = new float* [HiddenNum];
        BiasHidden = new float[HiddenNum];
        for (int i = 0; i < HiddenNum; ++i) {
            WeightHidden[i] = new float[InputNum];
            for (int k = 0; k < InputNum; ++k)
                WeightHidden[i][k] = (float)rand() / RAND_MAX - 0.5f;
            BiasHidden[i] = (float)rand() / RAND_MAX - 0.5f;
        }
        Output = new float[OutputNum];
        Hidden = new float[HiddenNum];
        HiddenDelta = new float[HiddenNum];
    }
    ~Mlp() {
        delete[] Output; delete[] Hidden; delete[] HiddenDelta;
        for (int i = 0; i < OutputNum; ++i) delete[] WeightOut[i];
        delete[] WeightOut; delete[] BiasOut;
        for (int i = 0; i < HiddenNum; ++i) delete[] WeightHidden[i];
        delete[] WeightHidden; delete[] BiasHidden;
    }

    int Compute(float* Input) {
        for (int h = 0; h < HiddenNum; ++h) {
            float sum = BiasHidden[h];
            for (int k = 0; k < InputNum; ++k)
                sum += WeightHidden[h][k] * Input[k];
            Hidden[h] = Sigmoid(sum);
        }
        for (int i = 0; i < OutputNum; ++i) {
            float sum = BiasOut[i];
            for (int h = 0; h < HiddenNum; ++h)
                sum += WeightOut[i][h] * Hidden[h];
            Output[i] = Sigmoid(sum);
        }
        float* max_pos = std::max_element(Output, Output + OutputNum);
        return (int)std::distance(Output, max_pos);
    }

    float Train(float* Input, int nClass) {
        Compute(Input);
        for (int k = 0; k < HiddenNum; ++k) HiddenDelta[k] = 0.f;
        float OutputDelta[10];
        float error = 0.f;
        for (int i = 0; i < OutputNum; ++i) {
            float target = (i == nClass) ? 1.f : 0.f;
            float e = target - Output[i];
            OutputDelta[i] = e * Output[i] * (1.f - Output[i]);
            error += e * e;
        }
        for (int h = 0; h < HiddenNum; ++h) {
            float back = 0.f;
            for (int i = 0; i < OutputNum; ++i)
                back += OutputDelta[i] * WeightOut[i][h];
            HiddenDelta[h] = Hidden[h] * (1.f - Hidden[h]) * back;
        }
        for (int i = 0; i < OutputNum; ++i) {
            for (int h = 0; h < HiddenNum; ++h)
                WeightOut[i][h] += LearningRate * OutputDelta[i] * Hidden[h];
            BiasOut[i] += LearningRate * OutputDelta[i];
        }
        for (int h = 0; h < HiddenNum; ++h) {
            for (int k = 0; k < InputNum; ++k)
                WeightHidden[h][k] += LearningRate * HiddenDelta[h] * Input[k];
            BiasHidden[h] += LearningRate * HiddenDelta[h];
        }
        return error;
    }
};

void MakeFeatures(std::vector<cv::Mat>& images,
    std::vector<std::vector<float>>& descriptorVector) {
    for (size_t idx = 0; idx < images.size(); ++idx) {
        cv::Mat& image = images[idx];
        cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
        cv::HOGDescriptor hog(cv::Size(32, 32), cv::Size(16, 16),
            cv::Size(8, 8), cv::Size(8, 8), 9);
        std::vector<float> descriptors;
        hog.compute(image, descriptors, cv::Size(32, 32), cv::Size(0, 0));
        descriptorVector.push_back(descriptors);
        image.convertTo(image, CV_32FC1);
        image /= 255.f;
    }
}

int main(int argc, char** argv) {
    srand((unsigned)time(0));
    // 기본 data 디렉터리: 인자 우선, 없으면 exe 옆의 cifar-10-batches-bin, 없으면 ./data/
    std::string dir = "data/";
    int EPOCH = 10;
    if (argc > 1) dir = std::string(argv[1]) + "/";
    if (argc > 2) EPOCH = std::atoi(argv[2]);

    if (argc <= 1) {
        // exe 폴더 확인
        std::filesystem::path exePath;
        char buf[MAX_PATH];
        GetModuleFileNameA(NULL, buf, MAX_PATH);
        exePath = std::filesystem::path(buf).parent_path();
        std::filesystem::path alt1 = exePath / "cifar-10-batches-bin";
        std::filesystem::path alt2 = exePath / "data" / "cifar-10-batches-bin";
        if (std::filesystem::exists(alt1)) {
            dir = (alt1.string() + "/");
        }
        else if (std::filesystem::exists(alt2)) {
            dir = (alt2.string() + "/");
        }
    }

    std::vector<cv::Mat> images;
    std::vector<int> labels, labels2;
    int c1 = ReadCifar(dir + "data_batch_1.bin", &images, &labels, &labels2, false);
    int c2 = ReadCifar(dir + "data_batch_2.bin", &images, &labels, &labels2, false);
    int c3 = ReadCifar(dir + "data_batch_3.bin", &images, &labels, &labels2, false);
    int c4 = ReadCifar(dir + "data_batch_4.bin", &images, &labels, &labels2, false);
    int c5 = ReadCifar(dir + "data_batch_5.bin", &images, &labels, &labels2, false);

    std::vector<std::vector<float>> descriptorVector;
    MakeFeatures(images, descriptorVector);

    std::vector<cv::Mat> t_images;
    std::vector<int> t_labels, t_labels2;
    int tc = ReadCifar(dir + "test_batch.bin", &t_images, &t_labels, &t_labels2, false);

    std::vector<std::vector<float>> t_descriptorVector;
    MakeFeatures(t_images, t_descriptorVector);

    if (images.empty() || t_images.empty()) {
        std::cout << "자료를 읽지 못했습니다. 다음 폴더를 확인하세요: " << dir << "\n";
        std::cout << "프로그램을 실행할 때 data 폴더 경로를 인자로 주거나, 실행파일 옆 cifar-10-batches-bin 폴더에 *.bin 파일을 두세요.\n";
        std::cout << "사용법: main.exe <data폴더경로> <epoch>\n";
        return -1;
    }

    Perceptron per1(32 * 32, 10, 0.2f), per2(324, 10, 0.2f);
    Mlp mlp1(32 * 32, 50, 10, 0.2f), mlp2(324, 50, 10, 0.2f);

    std::ofstream csv("results.csv");
    csv << "epoch,err_per_gray,err_per_hog,err_mlp_gray,err_mlp_hog,"
        << "acc_per_gray,acc_per_hog,acc_mlp_gray,acc_mlp_hog\n";

    for (int k = 0; k < EPOCH; ++k) {
        float error[4] = { 0 };
        for (int i = 0; i < (int)images.size(); ++i) {
            error[0] += per1.Train((float*)images[i].data, labels[i]);
            error[1] += per2.Train(descriptorVector[i].data(), labels[i]);
            error[2] += mlp1.Train((float*)images[i].data, labels[i]);
            error[3] += mlp2.Train(descriptorVector[i].data(), labels[i]);
        }
        int n = (int)images.size();
        printf("%d epoch, %lf %lf %lf %lf\n", k + 1,
            error[0] / n, error[1] / n, error[2] / n, error[3] / n);

        int trueCnt[4] = { 0 };
        for (int i = 0; i < (int)t_images.size(); ++i) {
            if (per1.Compute((float*)t_images[i].data) == t_labels[i]) trueCnt[0]++;
            if (per2.Compute(t_descriptorVector[i].data()) == t_labels[i]) trueCnt[1]++;
            if (mlp1.Compute((float*)t_images[i].data) == t_labels[i]) trueCnt[2]++;
            if (mlp2.Compute(t_descriptorVector[i].data()) == t_labels[i]) trueCnt[3]++;
        }
        int tn = (int)t_images.size();
        printf("%d epoch, %d, %.2f%% %d, %.2f%% %d, %.2f%% %d, %.2f%%\n", k + 1,
            trueCnt[0], 100.0 * trueCnt[0] / tn,
            trueCnt[1], 100.0 * trueCnt[1] / tn,
            trueCnt[2], 100.0 * trueCnt[2] / tn,
            trueCnt[3], 100.0 * trueCnt[3] / tn);

        csv << k + 1 << ','
            << error[0] / n << ',' << error[1] / n << ',' << error[2] / n << ',' << error[3] / n << ','
            << 100.0 * trueCnt[0] / tn << ',' << 100.0 * trueCnt[1] / tn << ','
            << 100.0 * trueCnt[2] / tn << ',' << 100.0 * trueCnt[3] / tn << '\n';
    }
    csv.close();
    std::cout << "\n결과가 results.csv 에 저장되었습니다.\n";
    return 0;
}
