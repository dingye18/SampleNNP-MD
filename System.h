#include <algorithm>
#include <iostream>
#include <string>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "NNPInter.h"
#include "Vec3.h"

void checkStatus(const tensorflow::Status& status);


class atom{
public:
    atom(unsigned int index, int elementIndex, string elementSymbol, double mass): index(index), elementIndex(elementIndex), mass(mass), elementSymbol(elementSymbol){
        inverseMass = 1.0/mass;
    }
    unsigned int index;
    double mass, inverseMass;
    string elementSymbol;
    unsigned int elementIndex;
};

class System{
public:
    System(unsigned int natoms): natoms(natoms){};
    vector<atom> atoms;
    vector<Vec3> positions;
    vector<Vec3> velocities;
    unsigned int natoms;
    vector<double> kinetic_energy, potential_energy, total_energy;
};