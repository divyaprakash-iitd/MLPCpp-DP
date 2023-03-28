/*!
 * \file CLayer.cpp
 * \brief Implementation of the Layer class to be used in the NeuralNetwork
 *      class
 * \author E.C.Bunschoten
 * \version 1.0.0
 */
#include "../include/CLayer.hpp"

#include <cstring>
using namespace std;

MLPToolbox::CLayer::CLayer() : CLayer(1) {}

MLPToolbox::CLayer::CLayer(unsigned long n_neurons)
    : number_of_neurons{n_neurons}, is_input{false} {
  neurons.resize(n_neurons);
  for (size_t i = 0; i < number_of_neurons; i++) {
    neurons[i].SetNumber(i + 1);
  }
}

void MLPToolbox::CLayer::SetNNeurons(unsigned long n_neurons) {
  if (number_of_neurons != n_neurons) {
    neurons.resize(n_neurons);
    for (size_t i = 0; i < number_of_neurons; i++) {
      neurons[i].SetNumber(i + 1);
    }
  }
}
