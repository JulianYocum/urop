//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
//
/// \file B2EventAction.cc
/// \brief Implementation of the B2EventAction class

#include "B2EventAction.hh"
#include "B2RootHandler.hh"

#include "G4Event.hh"
#include "G4EventManager.hh"
#include "G4TrajectoryContainer.hh"
#include "G4Trajectory.hh"
#include "G4ios.hh"

#include <iostream>
#include "B2TrackerHit.hh"

#include "TFile.h"
#include "TTree.h"

#include <vector>
#include <algorithm>

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

B2EventAction::B2EventAction()
: G4UserEventAction()
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

B2EventAction::~B2EventAction()
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void B2EventAction::BeginOfEventAction(const G4Event*)
{
    std::cout.precision(17);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void B2EventAction::EndOfEventAction(const G4Event* event)
{
  // get number of stored trajectories

  G4TrajectoryContainer* trajectoryContainer = event->GetTrajectoryContainer();
  G4int n_trajectories = 0;
  if (trajectoryContainer) n_trajectories = trajectoryContainer->entries();

  // periodic printing

  G4int eventID = event->GetEventID();

  if ( eventID < 100 || eventID % 100 == 0) {
      G4cout << ">>> Event: " << eventID  << G4endl;
  }
  /*
    if ( trajectoryContainer ) {
      G4cout << "    " << n_trajectories
             << " trajectories stored in this event." << G4endl;
    }
  */


  G4VHitsCollection* hc = event->GetHCofThisEvent()->GetHC(0);
  //G4cout << "    "
    //   << hc->GetSize() << " hits stored in this event" << G4endl;

  //std::cout << "    "
    //        << hc->GetSize() << " hits stored in this event" << std::endl;

  G4double total_Edep = 0;
  G4double final_KE = 0;
  G4ThreeVector final_Mvec;
  G4ThreeVector final_Pvec;

  std::vector<G4int> ids;
  G4int muons = 0;
  G4int electrons = 0;
  G4int positrons = 0;
  G4int gammas = 0;
  G4int other = 0;

  for(size_t i = 0; i < hc->GetSize(); i++)
  {
      G4VHit* hit = hc->GetHit(i);
      G4int id = ((B2TrackerHit *)hit)->GetTrackID();
      G4String name = ((B2TrackerHit *)hit)->GetName();
      G4double Edep =((B2TrackerHit *)hit)->GetEdep();
      total_Edep += Edep;
      //std::cout << Edep << std::endl;

      if (id == 1){
          final_KE = ((B2TrackerHit *)hit)->GetKE();
          final_Mvec = ((B2TrackerHit *)hit)->GetMvec();
          final_Pvec = ((B2TrackerHit *)hit)->GetPos();
      }

      if (!(std::find(ids.begin(), ids.end(), id) != ids.end())){
            if (name == "mu-"){
                muons++;
            }
            else if (name == "e-"){
                electrons++;
            }
            else if (name == "e+"){
                positrons++;
            }
            else if (name == "gamma"){
                gammas++;
            }
            else{
                other++;
            }
          ids.push_back(id);
      }
  }

  typedef struct {
     //Int_t ntrack,nseg,nvertex;
     //UInt_t flag;
     //Float_t temperature;
     Double_t Edep, KE_i, KE_f;
     //Float_t Mvec_ix, Mvec_iy, Mvec_iz;
     //Float_t Mvec_fx, Mvec_fy, Mvec_fz;
 } MUON_DATA;

 MUON_DATA muon_branch;

 typedef struct {
     Int_t Muons;
     Int_t Electrons;
     Int_t Positrons;
     Int_t Gammas;
     Int_t Other;
 } PARTICLES_DATA;

 PARTICLES_DATA particles_branch;

 G4PrimaryParticle *primary = event->GetPrimaryVertex(0)->GetPrimary(0);

 muon_branch.Edep = total_Edep;
 muon_branch.KE_i = primary->GetKineticEnergy();
 muon_branch.KE_f = final_KE;

 G4ThreeVector init_Mvec = primary->GetMomentumDirection();
 G4ThreeVector init_Pvec = event->GetPrimaryVertex(0)->GetPosition();

 auto *Mvec_i = new std::vector<Double_t>;
 auto *Mvec_f = new std::vector<Double_t>;

 *Mvec_i = {init_Mvec.x(), init_Mvec.y(), init_Mvec.z()};
 *Mvec_f = {final_Mvec.x(), final_Mvec.y(), final_Mvec.z()};

 auto *Pvec_i = new std::vector<Double_t>;
 auto *Pvec_f = new std::vector<Double_t>;

 *Pvec_i = {init_Pvec.getX(), init_Pvec.getY(), init_Pvec.getZ()};
 *Pvec_f = {final_Pvec.x(), final_Pvec.y(), final_Pvec.z()};
 /*
 muon_branch.Mvec_ix = Mvec_i.x();
 muon_branch.Mvec_iy = Mvec_i.y();
 muon_branch.Mvec_iz = Mvec_i.z();
 muon_branch.Mvec_fx = Mvec_f.x();
 muon_branch.Mvec_fy = Mvec_f.y();
 muon_branch.Mvec_fz = Mvec_f.z();
 */
/*
 muon_branch.Mvec_i[0] = Mvec_i.x();
 muon_branch.Mvec_i[1] = Mvec_i.y();
 muon_branch.Mvec_i[2] = Mvec_i.z();

 muon_branch.Mvec_f[0] = Mvec_f.x();
 muon_branch.Mvec_f[1] = Mvec_f.y();
 muon_branch.Mvec_f[2] = Mvec_f.z();
*/

 particles_branch.Muons = muons;
 particles_branch.Electrons = electrons;
 particles_branch.Positrons = positrons;
 particles_branch.Gammas = gammas;
 particles_branch.Other = other;


 /*
 std::cout << std::endl;
 //std::cout << "id: " << id << std::endl;
 //std::cout << "name: " << name << std::endl;
 std::cout << "Edep: " << muon_branch.Edep << std::endl;
 std::cout << "KE_i: " << muon_branch.KE_i << std::endl;
 std::cout << "KE_f: " << muon_branch.KE_f << std::endl;

 std::cout << std::endl;
 std::cout << "muons: " << muons << std::endl;
 std::cout << "electrons: " << electrons << std::endl;
 std::cout << "positrions: " << positrons << std::endl;
 std::cout << "gammas: " << gammas << std::endl;
 std::cout << "other: " << other << std::endl;
 */

 //std::cout << std::endl << "Edep:" << branch.Edep << std::endl;

 TTree * tree = roothandler->get_tree();
 tree->Print();

 /* try to open file to read */
 /*
 std::ifstream ifile;
 TFile * hfile;
 TTree * tree;

 ifile.open("geant4.root");
 if(ifile) {
   //ifile.close();
   //std::cout << std::endl << "reading TTree..." << std::endl;
   hfile = new TFile("geant4.root","update");

   tree = (TTree*)hfile->Get("T");
   tree->SetBranchAddress("muon",&muon_branch);
   tree->SetBranchAddress("Mvec_i", &Mvec_i);
   tree->SetBranchAddress("Mvec_f", &Mvec_f);
   tree->SetBranchAddress("Pvec_i", &Pvec_i);
   tree->SetBranchAddress("Pvec_f", &Pvec_f);
   //tree->SetBranchAddress("final", Mvec_f);
   tree->SetBranchAddress("particles",&particles_branch);
  } else {
   //ifile.close();
   hfile = new TFile("geant4.root","RECREATE","Geant4 Root Tree");
   tree = new TTree("T","An example of ROOT tree with a few branches");
   tree->Branch("muon",&muon_branch,"Edep/D:KE_i/D:KE_f/D");
   tree->Branch("Mvec_i", "std::vector<Double_t>", &Mvec_i, 32000, 0);
   tree->Branch("Mvec_f", "std::vector<Double_t>", &Mvec_f, 32000, 0);
   tree->Branch("Pvec_i", "std::vector<Double_t>", &Pvec_i, 32000, 0);
   tree->Branch("Pvec_f", "std::vector<Double_t>", &Pvec_f, 32000, 0);
   //tree->Branch("final", Mvec_f, "Mvec_f[3]/D");
   tree->Branch("particles", &particles_branch, "Muons/I:Electrons/I:Positrons/I:Gammas/I:Other/I");
  }
  */

 //std::cout << "here" << std::endl;

 tree->Fill();
 //tree->Print();




  /*
  std::cout << "total_Edep: " << total_Edep << std::endl;

  //G4cout << "end: ";
  //event->GetPrimaryVertex()->GetPrimary()->Print();
  std::cout << std::endl;
  std::cout << "Total Energy: " << event->GetPrimaryVertex(0)->GetPrimary(0)->GetTotalEnergy() << std::endl;
  std::cout << "Init Kinetic Energy: " << event->GetPrimaryVertex(0)->GetPrimary(0)->GetKineticEnergy() << std::endl;
  std::cout << "Init Vector: " << event->GetPrimaryVertex(0)->GetPrimary(0)->GetMomentumDirection() << std::endl << std::endl;
  std::cout << "Final Kinetic Energy: " << final_KE << std::endl;
  std::cout << "Final Vector: " <<  final_Mvec << std::endl;
  */




    //std::cout << "Kinetic Energy: " << event->GetPrimaryVertex()->GetPrimary()->GetKineticEnergy() << std::endl;


    //std::cout << hc->GetHit(0)->GetEdepID() << std::endl;
    //((Daughter *)ptr)->Missing()
    //hc->GetHit(0)->Print();

    //hc->GetHit(0)->
    //fRunAction->AddEdep(fEdep);
    /*
    int size = hc->GetSize();
    std::cout << "size: " << size << std::endl;

    for(int i=0; i < size; i++)
    {
        B2TrackerHit * hit = hc->GetHit (i);
        std::cout << "E: " << hit.GetEdep() << std::endl;
    }
    */
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
