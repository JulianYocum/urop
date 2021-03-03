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
/// \file B2PrimaryGeneratorAction.cc
/// \brief Implementation of the B2PrimaryGeneratorAction class

#include "B2PrimaryGeneratorAction.hh"

#include "G4LogicalVolumeStore.hh"
#include "G4LogicalVolume.hh"
#include "G4Box.hh"
#include "G4Sphere.hh"
#include "G4Event.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4SystemOfUnits.hh"
#include "G4ProcessManager.hh"

#include "Randomize.hh"

#include <cmath>
#include <iostream>
#include <math.h>
#include <fstream>

G4double log_uniform (G4double x)
{
    G4double a = 200;
    G4double b = 4000;

    return pow(a, 1-x) * pow(b,x);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

B2PrimaryGeneratorAction::B2PrimaryGeneratorAction()
 : G4VUserPrimaryGeneratorAction()
{
  //std::cout << "B2PrimaryGeneratorAction.cc: 50\n";

  G4int nofParticles = 1;
  fParticleGun = new G4ParticleGun(nofParticles);

  // default particle kinematic

  G4ParticleDefinition* particleDefinition
    = G4ParticleTable::GetParticleTable()->FindParticle("mu-");


 // std::cout << "processes: \n";
 // for (G4int i = 0; i < processlength; i++)
 // {
 //     std::cout << i << std::endl;
 //     std::cout << (*processes)[i]->GetProcessName() << std::endl;
 // }

  fParticleGun->SetParticleDefinition(particleDefinition);
  //fParticleGun->SetParticleEnergy(300*GeV);
  //std::cout << "B2PrimaryGeneratorAction.cc: 63\n";


   // G4ProcessManager* pmanager = particleDefinition->GetProcessManager();
   //
   // std::cout << "Here" << std::endl;
   //
   // if (pmanager == NULL)
   // {
   //     std::cout << "is null" << std::endl;
   // }
   //
   // G4ProcessVector* processes = pmanager->GetProcessList();
   //
   // std::cout << "Here" << std::endl;
   //
   // G4int processlength = processes->size();
   //
   // std::cout << "Here" << std::endl;
   // std::cout << processlength << std::endl;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

B2PrimaryGeneratorAction::~B2PrimaryGeneratorAction()
{
  delete fParticleGun;
}

G4ThreeVector GetRandVec()
{
    G4double x = G4RandGauss::shoot(0,1);//distribution(generator);
    G4double y = G4RandGauss::shoot(0,1);//distribution(generator);
    G4double z = G4RandGauss::shoot(0,1);//distribution(generator);

    G4double norm = 1 / std::pow(std::pow(x,2.0) + std::pow(y,2.0) + std::pow(z,2.0), .5);

    x *= norm;
    y *= norm;
    z *= norm;

    //std::cout << "x: " << x << std::endl << "y: " << y << std::endl << "z: " << z << std::endl;

    //std::cout << 50 * G4ThreeVector(x,y,z) << std::endl;

    return G4ThreeVector(x,y,z);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void B2PrimaryGeneratorAction::GeneratePrimaries(G4Event* anEvent)
{
  //std::cout << "B2PrimaryGeneratorAction.cc: 76\n";
  // This function is called at the begining of event

  // In order to avoid dependence of PrimaryGeneratorAction
  // on DetectorConstruction class we get world volume
  // from G4LogicalVolumeStore.

  G4double worldZHalfLength = 0;
  G4LogicalVolume* worldLV
    = G4LogicalVolumeStore::GetInstance()->GetVolume("World");
  G4Sphere* worldBox = NULL;
  if ( worldLV ) worldBox = dynamic_cast<G4Sphere*>(worldLV->GetSolid());
  if ( worldBox ) worldZHalfLength =  worldBox->GetRmax();//worldBox->GetZHalfLength();
  else  {
    G4cerr << "World volume of box not found." << G4endl;
    G4cerr << "Perhaps you have changed geometry." << G4endl;
    G4cerr << "The gun will be place in the center." << G4endl;
  }

  // Note that this particular case of starting a primary particle on the world boundary
  // requires shooting in a direction towards inside the world.

  //fParticleGun->SetParticlePosition(G4ThreeVector(0., 0., -worldZHalfLength));
  //unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  //std::default_random_engine generator (seed);
  //std::normal_distribution<double> distribution(0.0,1.0);

  /*
  G4ThreeVector pvec = 50/2 * sqrt(3) * GetRandVec();
  fParticleGun->SetParticlePosition(pvec);

  G4ThreeVector dvec;
  do{
      dvec = GetRandVec();
  } while(pvec.dot(dvec) >= 0);

  fParticleGun->SetParticleMomentumDirection(GetRandVec());
  fParticleGun->GeneratePrimaryVertex(anEvent);
  */
  G4ThreeVector p_vec = G4ThreeVector(G4UniformRand() * 50 - 25, G4UniformRand() * 50 - 25, 25);
  fParticleGun->SetParticlePosition(p_vec);

  G4ThreeVector v_vec = G4ThreeVector(G4RandGauss::shoot(0,1), G4RandGauss::shoot(0,1), -abs(G4RandGauss::shoot(0,1)));
  v_vec /= v_vec.mag();

  fParticleGun->SetParticleMomentumDirection(v_vec);
  fParticleGun->GeneratePrimaryVertex(anEvent);

  G4double e = log_uniform(G4UniformRand());
  fParticleGun->SetParticleEnergy(e*GeV);

  //std::cout << "B2PrimaryGeneratorAction.cc: 101\n";
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
