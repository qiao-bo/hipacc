//
// Copyright (c) 2018, University of Erlangen-Nuremberg
// Copyright (c) 2012, University of Erlangen-Nuremberg
// Copyright (c) 2012, Siemens AG
// Copyright (c) 2010, ARM Limited
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

//===--- ASTFuse.h - Kernel Fusion for the AST --------------------------===//
//
// This file implements the fusion and printing of the translated kernels.
//
//===--------------------------------------------------------------------===//

#ifndef _ASTFUSE_H_
#define _ASTFUSE_H_

#include <clang/AST/Attr.h>
#include <clang/AST/Type.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Sema/Ownership.h>
#include <llvm/ADT/SmallVector.h>

#include "hipacc/Analysis/KernelStatistics.h"
#include "hipacc/Analysis/HostDataDeps.h"
#include "hipacc/AST/ASTNode.h"
#include "hipacc/AST/ASTTranslate.h"
#include "hipacc/Config/CompilerOptions.h"
#include "hipacc/Device/Builtins.h"
#include "hipacc/DSL/ClassRepresentation.h"
#include "hipacc/Vectorization/SIMDTypes.h"

#include <functional>
#include <queue>
#include <errno.h>
#include <fcntl.h>

#ifdef _WIN32
# include <io.h>
# define popen(x,y) _popen(x,y)
# define pclose(x)  _pclose(x)
# define fsync(x)
#else
# include <unistd.h>
#endif

//#define PRINT_DEBUG

namespace clang {
namespace hipacc {

class ASTFuse {
  private:
    static const bool DEBUG;

    ASTContext &Ctx;
    DiagnosticsEngine &Diags;
    hipacc::Builtin::Context &builtins;
    CompilerOptions &compilerOptions;
    HipaccDevice targetDevice;
    PrintingPolicy Policy;
    HostDataDeps *dataDeps;

    // variables for all fusible kernel lists
    SmallVector<std::list<HipaccKernel *>, 16> vecFusibleKernelLists;
    std::map<HipaccKernel *, std::string> fusedKernelNameMap;
    std::map<HipaccKernel *, std::tuple<unsigned, unsigned>> fusedLocalKernelMemorySizeMap;
    SmallVector<std::string, 16> fusedFileNamesAll;

    // variables per fusible kernel lists
    FunctionDecl *curFusedKernelDecl;
    SmallVector<Stmt *, 16> curFusedKernelBody;
    std::string fusedKernelName;
    std::string fusedFileName;
    std::map<std::string, HipaccKernel *> FuncDeclParamKernelMap;
    std::map<std::string, FieldDecl *> FuncDeclParamDeclMap;
    std::map<HipaccKernel *, std::tuple<unsigned, unsigned>> localKernelMemorySizeMap;
    std::tuple<unsigned, unsigned> localKernelMaxAccSizeUpdated;
    unsigned fusionRegVarCount;
    SmallVector<VarDecl *, 16> fusionRegVarDecls;
    SmallVector<Stmt *, 16> fusionRegSharedStmts;
    unsigned fusionIdxVarCount;
    SmallVector<VarDecl *, 16> fusionIdxVarDecls;

    enum SubListPosition {
      Source,
      Intermediate,
      Destination,
      Undefined
    };

    struct FusionTypeTags {
      SubListPosition Point2PointLoc = Undefined;
      SubListPosition Local2PointLoc = Undefined;
      SubListPosition Point2LocalLoc = Undefined;
      SubListPosition Local2LocalLoc = Undefined;
      FusiblePartitionBlock::PatternType patternType;

      FusionTypeTags(FusiblePartitionBlock::PatternType patternType) : patternType(patternType) {}
    };

    struct KernelListLocation {
      // The location of the block in a set of partitionBlockNames
      unsigned blockLocation;

      // The location of the respective kernel list in a partitionBlockNames
      unsigned listLocation;
    };

    std::map<HipaccKernel *, FusionTypeTags *> FusibleKernelSubListPosMap;
    std::map<std::string, KernelListLocation> FusibleKernelBlockLocation;
    std::vector<std::list<HipaccKernel*> *> fusibleKernelSet;

    // member functions
    void setFusedKernelConfiguration(std::list<HipaccKernel *> *l);
    void printFusedKernelFunction(std::list<HipaccKernel *> *l);
    void HipaccFusion(std::list<HipaccKernel *> *l);
    void initKernelFusion();
    FunctionDecl *createFusedKernelDecl(std::list<HipaccKernel *> *l);
    void insertPrologFusedKernel();
    void insertEpilogFusedKernel();
    void createReg4FusionVarDecl(QualType QT);
    void createIdx4FusionVarDecl();
    void createGidVarDecl();
    void markKernelPositionSublist(std::list<HipaccKernel *> *l);
    void recomputeMemorySizeLocalFusion(std::list<HipaccKernel *> *l);

    const FusiblePartitionBlock& getPartitionBlockFor(std::list<HipaccKernel *> *l) {
      hipacc_require((!l->empty()), "There is no fusion type for empty lists.");

      auto fusibleBlocks = dataDeps->getFusiblePartitionBlocks();
      auto block = fusibleBlocks.end();

      for (auto k : *l) {
        auto innerBlock = FusiblePartitionBlock::findForKernel(k, fusibleBlocks);
        if (block != fusibleBlocks.end()) {
          hipacc_require((block == innerBlock), "The given kernel list contains kernels of distinct partition blocks.");
        } else {
          block = innerBlock;
        }
      }

      hipacc_require(block != fusibleBlocks.end(), "The given kernel list did not correspond to a partition block.");
      return *block;
    }

  public:
    ASTFuse(ASTContext& Ctx, DiagnosticsEngine &Diags, hipacc::Builtin::Context &builtins,
        CompilerOptions &options, PrintingPolicy Policy, HostDataDeps *dataDeps) :
      Ctx(Ctx),
      Diags(Diags),
      builtins(builtins),
      compilerOptions(options),
      targetDevice(options),
      Policy(Policy),
      dataDeps(dataDeps),
      curFusedKernelDecl(nullptr),
      fusedKernelName(""),
      fusedFileName(""),
      fusionRegVarCount(0),
      fusionIdxVarCount(0)
      {
        unsigned nFusibleKernelBlockLocations = 0;
        for (const auto& fusibleBlock : dataDeps->getFusiblePartitionBlocks()) { // block level
          if (!fusibleBlock.isPatternFusible()) {
            continue;
          }

          unsigned KernelVecID = 0;
          for (const auto& part : fusibleBlock.getParts()) {              // vector level
            KernelListLocation pos = {
              nFusibleKernelBlockLocations,
              KernelVecID
            };

            auto nam = part.front().getName();
            bool locExists = FusibleKernelBlockLocation.find(nam) != FusibleKernelBlockLocation.end();
            hipacc_require(!locExists, "Kernel lists cannot be added twice");

            FusibleKernelBlockLocation[nam] = pos;
            KernelVecID++;
          }
          // create a list for each partion block
          std::list<HipaccKernel*> *list = new std::list<HipaccKernel*>;
          fusibleKernelSet.push_back(list);
          nFusibleKernelBlockLocations++;
        }
      }

    // Called by Rewriter
    bool parseFusibleKernel(HipaccKernel *K);
    SmallVector<std::string, 16> getFusedFileNamesAll() const;
    bool isSrcKernel(HipaccKernel *K);
    bool isDestKernel(HipaccKernel *K);
    HipaccKernel *getProducerKernel(HipaccKernel *K);
    std::string getFusedKernelName(HipaccKernel *K);
    unsigned getNewYSizeLocalKernel(HipaccKernel *K);

		friend std::ostream& operator<<(std::ostream &os, SubListPosition &slpos) {
			switch (slpos) {
			case Source:
					os << "Source";
					break;
			case Intermediate:
					os << "Intermediate";
					break;
			case Destination:
					os << "Destination";
					break;
			case Undefined:
					os << "Undefined";
			}
			return os;
		}
};

} // namespace hipacc
} // namespace clang

#endif  // _ASTFUSE_H_

// vim: set ts=2 sw=2 sts=2 et ai:

