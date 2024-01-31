#ifndef __DATA_HANDLER_H
#define __DATA_HANDLER_H

#include <fstream>
#include "stdint.h"
#include "data.hpp"
#include <vector>
#include <string>
#include <map>
#include <unordered_set>


class data_handler
{
  std::vector<data *> *data_array; // all of the data pre-split
  std::vector<data *> *training_data;
  std::vector<data *> *test_data;
  std::vector<data *> *validation_data;

  int num_classers;
  int feature_vector_size;
  std::map<uint8_t, int> class_map;

  const double TRAIN_SET_PERCENT = 0.75; // very large percentage of our data
  const double TEST_SET_PERCENT = 0.20; // smaller portion of our data
  const double VALIDATION_PERCENT = 0.05; //small amount of data to validate that our model is trained to our liking

  public:
    data_handler(); //constructor
    ~data_handler(); //destructor

    void read_feature_vector(std::string path);
    void read_feature_label(std::string path);
    void split_data();
    void count_classes();

    uint32_t convert_to_little_endian(const unsigned char* bytes); //files are stored in the MSB first (high endian), this function should only by used by intel processors

    std::vector<data *> * get_training_data();
    std::vector<data *> * get_test_data();
    std::vector<data *> * get_validation_data();
};

#endif
