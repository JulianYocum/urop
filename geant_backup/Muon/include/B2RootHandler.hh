#ifndef __B2_ROOT_HANDLER_HH__
#define __B2_ROOT_HANDLER_HH__

#include "TFile.h"
#include "TTree.h"
#include <fstream>
#include <iostream>

class RootHandler {
    std::ifstream * ifile;
    TFile * hfile;
    TTree * tree;

    public:
        RootHandler();
        void set_tree();
        TTree * get_tree();
        TFile * get_hfile();
};

extern RootHandler * roothandler = new RootHandler();

#endif
