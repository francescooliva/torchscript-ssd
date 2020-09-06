#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <fstream>
#include <cassert>
#include <chrono>
#include <exception>
#include <string>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/utility.hpp>


static void print_tensor_shape(const torch::Tensor &t)
{
  std::vector<int64_t> shape;

  for (int i = 0; i < t.dim(); ++i) {
    shape.push_back(t.size(i));
  }
  std::cout << "Tensor " << t.name() << " shape: {" << shape << "}" << std::endl;
}

static torch::Tensor area_of(const torch::Tensor &top_left,
                             const torch::Tensor &bottom_right)
{
  torch::Tensor hw = bottom_right - top_left;
  hw.clamp_min(torch::Scalar(0.f));

  return hw.select(-1, 0) * hw.select(-1, 1);
}

static torch::Tensor iou_of(const torch::Tensor &boxes0,
                            const torch::Tensor &boxes1,
                            float eps = 1.e-5f)
{
  torch::Tensor overlap_top_left = torch::max(boxes0.slice(-1, 0, 2),
                                              boxes1.slice(-1, 0, 2));
  torch::Tensor overlap_bottom_right = torch::min(boxes0.slice(-1, 2),
                                                  boxes1.slice(-1, 2));
  torch::Tensor overlap_area = area_of(overlap_top_left, overlap_bottom_right);
  torch::Tensor area0 = area_of(boxes0.slice(-1, 0, 2), boxes0.slice(-1, 2));
  torch::Tensor area1 = area_of(boxes1.slice(-1, 0, 2), boxes1.slice(-1, 2));

  return overlap_area / (area0 + area1 - overlap_area + torch::Scalar(eps));
}

torch::Tensor hard_nms(torch::Tensor t, float iou_threshold, int top_k,
                       int candidates)
{
  using namespace std;

  torch::Tensor scores = t.select(1, -1);
  torch::Tensor boxes = t.slice(1, 0, -1);

  torch::Tensor indices = scores.argsort(-1, true).slice(0, 0, candidates);

  std::vector<int64_t> picked;
  while (0 < indices.numel()) {
    torch::Tensor current = indices.select(0, 0);
    picked.push_back(current.item().toLong());
    if ((0 < top_k && static_cast<int>(picked.size())) ||
        1 == indices.numel()) break;
    torch::Tensor current_box = boxes.select(0, current.item().toLong());
    indices = indices.narrow(0, 1, indices.numel() - 1);
    torch::Tensor remained = boxes.index_select(0, indices);

    torch::Tensor iou = iou_of(remained, current_box.unsqueeze(0));
    indices = indices.masked_select(iou <= iou_threshold);
  }

  torch::Tensor t_picked =
    torch::from_blob(picked.data(), {static_cast<int64_t>(picked.size())},
                     torch::TensorOptions(torch::ScalarType::Long));

  return t.index_select(0, t_picked);
}

static const std::string keys = 
  "{ help h   | | Print help message. }"
  "{ torch-script s  | ../models/mobilenet-v1-ssd-mp-0_675.pt | Path to the TorchScript file.}"
  "{ labels l | ../models/voc-model-labels.txt | labels file}"
  "{ backbone b | mb1 | mb1 or vgg16}"
  "{ probability p | 0.5 | object score threshold}"
  "{ video v  | ../videos/test.mp4 | video name, for webcam on Linux use /dev/video0}";

int main(int argc, const char* argv[])
{
  cv::CommandLineParser parser(argc, argv, keys);
  if (parser.has("help")){
    parser.printMessage();
    return 0;  
  }

  std::string ts_filename = parser.get<std::string>("torch-script");
  std::cout << ts_filename << std::endl;
  std::string label_filename = parser.get<std::string>("labels");
  std::string video_name = parser.get<std::string>("video");
   std::cout << video_name << std::endl;
  std::string backbone = parser.get<std::string>("backbone");
  assert(!video_name.empty());
  assert(!label_filename.empty());
  int network_resolution = 300;
  int candidate_size = 200;
  int top_k = 10;
  float probability_threshold = parser.get<float>("probability");
  float iou_threshold = 0.3f;
  torch::DeviceType device_type;

  std::cout << "cuDNN: "
     << (torch::cuda::cudnn_is_available() ? "Yes" : "No") << std::endl;
  std::cout << "CUDA: " << (torch::cuda::is_available() ?  "Yes" : "No") << std::endl;
  if (torch::cuda::is_available() ) {
    device_type = torch::kCUDA;
  } else {
    device_type = torch::kCPU;
  }
  torch::Device device(device_type);

  std::vector<std::string> labels;
  std::ifstream file(label_filename);
  std::string str;
  while (std::getline(file, str))
    labels.push_back(str);

  torch::jit::script::Module module;
  try {
    module = torch::jit::load(ts_filename);
    std::cout << "Loaded TorchScript " << ts_filename << std::endl;
  } catch (const c10::Error &e) {
    std::cerr << "Error loading the TorchScript " << ts_filename << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Start inferencing ..." << std::endl;

  cv::Mat input_image, resized_image;
  cv::VideoCapture cap(video_name);
  while (cap.read(input_image))
  {
      std::cout << "Original image size [width, height] = [" << input_image.cols
          << ", " << input_image.rows << "]" << std::endl;
      cv::cvtColor(input_image, resized_image, cv::COLOR_BGR2RGB);
      cv::resize(resized_image, resized_image,
          cv::Size(network_resolution, network_resolution));

      cv::Mat img_float;
      torch::Tensor tensor_image;
      if(backbone == "vgg16"){
        resized_image.convertTo(img_float, CV_32FC3);
        img_float -= cv::Scalar(103.939, 116.779, 123.68);
        tensor_image = torch::from_blob(resized_image.data, {1, network_resolution, network_resolution, 3}, torch::kByte)
        .to(device).permute({0, 3, 1, 2});
        tensor_image = tensor_image.toType(torch::kFloat);
      }  
      else if(backbone == "mb1"){
        resized_image.convertTo(img_float, CV_32F, 1.0 / 128, -127.0 / 128);
        tensor_image = torch::from_blob(img_float.data, {1, network_resolution, network_resolution, 3})
        .to(device).permute({0, 3, 1, 2});
      }


      auto start = std::chrono::high_resolution_clock::now();

      std::vector<torch::jit::IValue> inputs;
      inputs.push_back(tensor_image);

      auto output = module.forward(inputs).toTuple()->elements();

      auto end = std::chrono::high_resolution_clock::now();

      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

      // It should be known that it takes longer time at first time
      std::cout << "Inference done in " << duration.count() << " ms" << std::endl;

      torch::Tensor scores = output[0].toTensor().select(0, 0).to(torch::kCPU);
      torch::Tensor boxes = output[1].toTensor().select(0, 0).to(torch::kCPU);

      print_tensor_shape(scores);
      print_tensor_shape(boxes);

      torch::Tensor picked_box_probs = torch::empty({ 0 });
      std::vector<int> picked_labels;
      for (int class_index = 1; class_index < scores.size(1); ++class_index)
      {
          torch::Tensor mask = scores.select(1, class_index) > probability_threshold;
          torch::Tensor prob = scores.select(1, class_index).masked_select(mask);
          torch::Tensor selected_boxes = boxes.index_select(0, mask.nonzero().squeeze());

          if (0 == selected_boxes.size(0)) continue;

          std::cout << "Class index [" << class_index << "]: "
               << labels.at(class_index) << std::endl;
          torch::Tensor box_prob = torch::cat({ selected_boxes, prob.reshape({-1, 1}) }, 1);
          box_prob = hard_nms(box_prob, iou_threshold, top_k, candidate_size);
          picked_box_probs = torch::cat({ picked_box_probs, box_prob }, 0);
          picked_labels.insert(picked_labels.end(), box_prob.size(0), class_index);
      }

      print_tensor_shape(picked_box_probs);
      if (0 == picked_box_probs.size(0))
      {
          std::cout << "No object detected." << std::endl;
          continue;
      }

      auto ra = picked_box_probs.accessor<float, 2>();
      for (int i = 0; i < ra.size(0); ++i)
      {
          ra[i][0] *= input_image.cols;
          ra[i][1] *= input_image.rows;
          ra[i][2] *= input_image.cols;
          ra[i][3] *= input_image.rows;

          cv::rectangle(input_image, cv::Point(ra[i][0], ra[i][1]),
              cv::Point(ra[i][2], ra[i][3]), cv::Scalar(255, 255, 0), 4);
          std::ostringstream oss;
          oss.precision(3);
          oss << labels.at(picked_labels.at(i)) << ": " << ra[i][4];
          cv::putText(input_image, oss.str(), cv::Point(ra[i][0] + 20, ra[i][1] + 40),
              cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 255), 2);
          cv::imshow("", input_image);
          cv::waitKey(1);
      }
  }



  return EXIT_SUCCESS;
}

