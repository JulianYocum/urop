#include "TFile.h"
#include "TTree.h"
#include <fstream>
#include <iostream>
#include "B2RootHandler.hh"

// class RootHandler {
//     std::ifstream * ifile;
//     TFile * hfile;
//     TTree * tree;
//     public:
//         //RootHandler();
//         void set_tree();
//         TTree * get_tree();
//         TFile * get_hfile();
// };

// RootHandler::RootHandler(){
// ifile=0;
// tree=0;
// hfile=0;
// }

RootHandler::RootHandler() : ifile(0), hfile(0), tree(0) {}

TTree * RootHandler::get_tree(){
    return tree;
}

TFile * RootHandler::get_hfile(){
    return hfile;
}

void RootHandler::set_tree (){

    std::cout << "in set_tree:" << std::endl;
    /*
    if (!(ifile))
    {
        std::cout << "ifile is NULL" << std::endl;
    }
    std::cout << "here1" << std::endl;

    if (ifile->is_open()){
        std::cout << "FILE OPEN" << std::endl;
    }

    std::cout << "here2" << std::endl;


    ifile->open("geant4.root");
    std::cout << "here3" << std::endl;

    */

    // if(*ifile) {
    //   hfile = new TFile("geant4.root","update");
    //   tree = (TTree*)hfile->Get("T");
    //
    //  } else {
                        //ifile.close();
      hfile = new TFile("geant4.root","RECREATE","Geant4 Root Tree");
      tree = new TTree("T","An example of ROOT tree with a few branches");

    //}
}
