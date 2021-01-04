#include "System.h"
#include <c_api.h>
#include <fstream>
#include <typeinfo>

using namespace std;
using namespace tensorflow;

void checkStatus(const tensorflow::Status& status) {
  if (!status.ok()) {
    std::cout << status.ToString() << std::endl;
    exit(1);
  }
}

#define VALUETYPE float

unsigned int natoms = 12;
unsigned int nghost = 0;
int nsteps = 1000;
// Force and energy coefficients are used for units transformation from {eV/A, eV} to {kJ/(mol * nm), kJ/mol}.
double dener = 0., forceUnitCoeff = 964.8792534459, energyUnitCoeff = 96.48792534459;
double delta_T = 0.0001; // 0.0001 picoseconds here.

vector<VALUETYPE> dcoord = vector<VALUETYPE>(natoms * 3);
vector<int> dtype = {0,1,1,0,1,1,0,1,1,0,1,1};
vector<VALUETYPE> dbox = {20.0,0.0,0.0,0.0,20.0,0.0,0.0,0.0,20.0};
vector<VALUETYPE> dforce = vector<VALUETYPE>(natoms * 3, 0.);
vector<VALUETYPE> dvirial = vector<VALUETYPE>(9, 0.);
vector<Vec3> xPrime = vector<Vec3>(natoms, Vec3());


void update(double delta_T, System& system, int step){
    // Update the velocities and positions of each particle with Verlet Integrator.
    // Get new velocities first.
    for(int ii=0; ii < natoms; ii++){
        for(int jj=0; jj < 3; jj++){
            system.velocities[ii][jj] += static_cast<double>(dforce[ii * 3 + jj]) * forceUnitCoeff * system.atoms[ii].inverseMass * delta_T;
            xPrime[ii][jj] = system.positions[ii][jj] + system.velocities[ii][jj] * delta_T;
        }
    }
    // Update the positions
    for(int ii=0; ii < natoms; ii++){
        for(int jj = 0; jj < 3; jj++){
            system.velocities[ii][jj] = 1.0/delta_T * (xPrime[ii][jj] - system.positions[ii][jj]);
            system.positions[ii][jj] = xPrime[ii][jj];
        }
    }

    // Calculate the kinetic energy.
    vector<Vec3> shiftedVelocity = vector<Vec3>(natoms, Vec3());
    double kinetic = 0.;
    for(int ii = 0; ii < natoms; ii ++){
        for(int jj = 0; jj < 3; jj++){
            shiftedVelocity[ii][jj] = system.velocities[ii][jj] + static_cast<double>(dforce[ii * 3 +jj]) * 0.5 * delta_T * system.atoms[ii].inverseMass;
            kinetic +=  system.atoms[ii].mass * shiftedVelocity[ii][jj] * shiftedVelocity[ii][jj];
        }
    }
    kinetic = 0.5 * kinetic;
    system.kinetic_energy.push_back(kinetic);
    system.total_energy.push_back(kinetic + dener * energyUnitCoeff);
}

void step(double delta_T, NNPInter nnp_inter, System& system, int step){
    for(int ii=0; ii < natoms; ii++){
        // Multiply 0.1 for unit transformation from nanometers to angstrom.
        dcoord[ii * 3 + 0] = static_cast<float>(system.positions[ii][0]) * 10;
        dcoord[ii * 3 + 1] = static_cast<float>(system.positions[ii][1]) * 10;
        dcoord[ii * 3 + 2] = static_cast<float>(system.positions[ii][2]) * 10;
    }
    nnp_inter.compute(dener, dforce, dvirial, dcoord, dtype, dbox, nghost);
    system.potential_energy.push_back(dener * energyUnitCoeff);
    update(delta_T, system, step);
}


int main(){
    string model = "../model/lw_pimd.v1.pb";
    string OP_library  = "/home/dingye/.local/deepmd-kit-1.2.0/lib/libdeepmd_op.so";
    TF_Status* LoadOpStatus = TF_NewStatus();
    TF_LoadLibrary(OP_library.c_str(), LoadOpStatus);
    TF_DeleteStatus(LoadOpStatus);         
    cout.precision(10);
    
    
    double init_coords[] = {10.543000221252441, 14.571999549865723,7.9380002021789551,10.170000076293945,15.211000442504883,7.3270001411437988,11.420999526977539,14.894000053405762,8.116999626159668,3.4600000381469727,6.3179998397827148,1.784000039100647,2.7950000762939453,6.4429998397827148,1.1039999723434448,3.5339999198913574,5.3730001449584961,1.878000020980835,1.3240000009536743,14.984000205993652,5.8090000152587891,1.6160000562667847,15.803000450134277,6.2129998207092285,0.47200000286102295,15.189999580383301,5.434999942779541,3.9010000228881836,18.275999069213867,0.55199998617172241,3.684999942779541,18.003000259399414,1.4450000524520874,3.5929999351501465,17.562999725341797,0.0010000000474974513};
    
    NNPInter nnp_inter = NNPInter(model, 0);
    System sys = System(natoms);

    for(unsigned int ii = 0; ii < natoms; ii ++){
        string symbol = "";
        double mass = 0.;
        if(dtype[ii] == 0){
            symbol = "O";
            mass = 15.999;
        }
        else if(dtype[ii] == 1){
            symbol = "H";
            mass = 1.00784;
        }    
        sys.atoms.push_back(atom(ii, dtype[ii], symbol, mass));
        sys.positions.push_back(Vec3(init_coords[ii * 3 + 0] * 0.1, init_coords[ii * 3 + 1] * 0.1, init_coords[ii * 3 + 2] * 0.1));
        sys.velocities.push_back(Vec3(0., 0., 0.));
    }
    /*
    sys.kinetic_energy = vector<double>(steps, 0.);
    sys.potential_energy = vector<double>(steps, 0.);
    sys.total_energy = vector<double>(steps, 0.);
    */
   
    //nnp_inter.compute(dener, dforce, dvirial, dcoord, dtype, dbox, nghost);

    for(int ii = 0; ii < nsteps; ii ++){
        step(delta_T, nnp_inter, sys, ii);
        cout<< ii + 1 << " step (unit is kJ/mol): "<<endl;
        cout<<"Kinetic Energy: "<<sys.kinetic_energy.back()<<" ";
        cout<<"Potential Energy: "<<sys.potential_energy.back()<<" ";
        cout<<"Total Energy: "<<sys.total_energy.back()<<endl;
        cout<<"----------------------"<<endl;
    }

    return 0;
}

