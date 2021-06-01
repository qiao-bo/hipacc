//
// Copyright (c) 2018, University of Erlangen-Nuremberg
// Copyright (c) 2014, Saarland University
// Copyright (c) 2014, University of Erlangen-Nuremberg
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

//===------------- HostDataDeps.h - Track data dependencies ---------------===//
//
// This file implements tracking of data dependencies to provide information
// for optimization such as kernel fusion
//
//===----------------------------------------------------------------------===//

#ifndef _HOSTDATADEPS_H_
#define _HOSTDATADEPS_H_

#include <clang/AST/ASTContext.h>
#include <clang/AST/StmtVisitor.h>
#include <clang/AST/Attr.h>
#include <clang/AST/Type.h>
#include <clang/Analysis/AnalysisDeclContext.h>
#include <clang/Analysis/Analyses/PostOrderCFGView.h>
#include <clang/Basic/Diagnostic.h>

#include "hipacc/Analysis/KernelStatistics.h"
#include "hipacc/Device/TargetDescription.h"
#include "hipacc/Device/Builtins.h"
#include "hipacc/Config/CompilerOptions.h"
#include "hipacc/DSL/CompilerKnownClasses.h"
#include "hipacc/DSL/ClassRepresentation.h"
#include "hipacc/AST/ASTNode.h"

#include <vector>
#include <list>
#include <tuple>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <unordered_set>

//#define PRINT_DEBUG

namespace clang {
namespace hipacc {

class HostDataDeps;

class DependencyTracker : public StmtVisitor<DependencyTracker> {
  private:
    static const bool DEBUG;

    ASTContext &Context;
    PrintingPolicy &Policy;
    CompilerKnownClasses &compilerClasses;
    HostDataDeps &dataDeps;

    llvm::DenseMap<ValueDecl *, HipaccAccessor *> accDeclMap_;
    llvm::DenseMap<ValueDecl *, HipaccImage *> imgDeclMap_;
    llvm::DenseMap<ValueDecl *, HipaccIterationSpace *> iterDeclMap_;
    llvm::DenseMap<ValueDecl *, HipaccBoundaryCondition *> bcDeclMap_;
    llvm::DenseMap<ValueDecl *, HipaccMask *> maskDeclMap_;

    std::vector<VarDecl *> visitedKernelDecl_;

  public:
    DependencyTracker(ASTContext &Context, PrintingPolicy &Policy,
                      AnalysisDeclContext &analysisContext,
                      CompilerKnownClasses &compilerClasses,
                      HostDataDeps &dataDeps)
        : Context(Context), Policy(Policy), compilerClasses(compilerClasses), dataDeps(dataDeps) {

      if (DEBUG) std::cout << "Tracking data dependencies:" << std::endl;
      PostOrderCFGView *POV = analysisContext.getAnalysis<PostOrderCFGView>();
      for (auto it=POV->begin(), ei=POV->end(); it!=ei; ++it) {
        // apply the transfer function for all Stmts in the block.
        const CFGBlock *block = static_cast<const CFGBlock*>(*it);
        for (auto it = block->begin(), ei = block->end(); it != ei; ++it) {
          const CFGElement &elem = *it;
          if (!elem.getAs<CFGStmt>()) continue;
          const Stmt *S = elem.castAs<CFGStmt>().getStmt();
          this->Visit(const_cast<Stmt*>(S));
        }
      }
      if (DEBUG) std::cout << std::endl;
    }

    void VisitDeclStmt(DeclStmt *S);
    void VisitCXXMemberCallExpr(CXXMemberCallExpr *E);
    void VisitCXXOperatorCallExpr(CXXOperatorCallExpr *E);

    std::string convertToString(Stmt *from) {
      hipacc_require(from != nullptr, "Expected non-null Stmt");
      std::string SS;
      llvm::raw_string_ostream S(SS);
      from->printPretty(S, nullptr, Policy);
      return S.str();
    }
};

class FusiblePartitionBlock;

class HostDataDeps : public ManagedAnalysis {
  friend class DependencyTracker;
  friend class FusiblePartitionBlock;

  private:
    static const bool DEBUG;

    // forward declarations
    class Node;
    class Space;
    class Process;
    class Accessor;
    class Image;
    class Mask;
    class IterationSpace;
    class BoundaryCondition;
    class Kernel;

    // member variables
    CompilerKnownClasses compilerClasses;
    CompilerOptions *compilerOptions;

    llvm::DenseMap<ValueDecl *, Accessor *> accMap_;
    llvm::DenseMap<ValueDecl *, Image *> imgMap_;
    llvm::DenseMap<ValueDecl *, Mask *> maskMap_;
    llvm::DenseMap<ValueDecl *, IterationSpace *> iterMap_;
    llvm::DenseMap<ValueDecl *, BoundaryCondition *> bcMap_;
    llvm::DenseMap<ValueDecl *, Kernel *> kernelMap_;
    std::map<std::string, Process *> processMap_;
    std::map<std::string, Space *> spaceMap_;
    std::vector<Space*> spaces_;
    std::vector<Process*> processes_;
    std::vector<Process*> fusibleProcesses_;
    llvm::DenseMap<RecordDecl *, HipaccKernelClass *> KernelClassDeclMap;
    std::map<Process *, bool> processVisitorMap_;
    std::map<Process *, std::vector<std::string>> visitedKernelDeclNameMap_;

    std::map<std::string, std::set<std::string>> graphNodeDepMap_;
    std::map<std::string, std::string> graphImgMemcpyNodeMap_;

    // application graph representations
    std::map<Process *, std::list<Process*> *> FusibleKernelListsMap;
    std::map<Process *, unsigned> FusibleProcessListSizeFinalMap;
    std::vector<std::list<Process*> *> vecFusibleKernelLists;
    using partitionBlock = std::vector<std::list<Process*> *>;
    partitionBlock applicationGraph;
    using partitionBlockNames = std::vector<std::list<std::string>>;
    std::set<FusiblePartitionBlock> fusiblePartitionBlocks;
    using edgeWeight = std::map<std::pair<Process *, Process *>, unsigned>;
    edgeWeight edgeWeightMap_;

    // parameters for prediction model
    // TODO, move to device*.h
    const unsigned GAMMA = 1;
    const unsigned EPSILON = 1;
    const unsigned TG = 800;
    const unsigned TS = 2;
    const unsigned CALU = 2;
    const unsigned CSFU = 16;
    const unsigned CMS = 2;
    float CMSf = 1.5;

    // inner class definitions
    class IterationSpace {
      private:
        HipaccIterationSpace *iter;
        Image *image;

      public:
        IterationSpace(HipaccIterationSpace *iter, Image *image)
            : iter(iter), image(image) {
        }

        std::string getName() {
          return iter->getName();
        }

        Image *getImage() {
          return image;
        }
    };

    class Accessor {
      private:
        HipaccAccessor *acc;
        Image *image;
        Space *space;

      public:
        Accessor(HipaccAccessor *acc, Image *image)
            : acc(acc), image(image), space(nullptr) {
        }

        Space *getSpace() {
          return space;
        }

        void setSpace(Space *space) {
          this->space = space;
        }

        std::string getName() {
          return acc->getName();
        }

        unsigned getSizeX() {
          return acc->getSizeX();
        }

        unsigned getSizeY() {
          return acc->getSizeY();
        }

        Image *getImage() {
          return image;
        }
    };

    class BoundaryCondition {
      private:
        HipaccBoundaryCondition *bc;
        Image* img;

      public:
        BoundaryCondition(HipaccBoundaryCondition *bc, Image* img)
            : bc(bc), img(img) {
        }

        Image* getImage() {
          return img;
        }
    };

    class Image {
      private:
        HipaccImage *img;

      public:
        explicit Image(HipaccImage *img)
            : img(img) {
        }

        std::string getName() {
          return img->getName();
        }
    };

    class Mask {
      private:
        HipaccMask *mask;

      public:
        explicit Mask(HipaccMask *mask)
            : mask(mask) {
        }

        std::string getName() {
          return mask->getName();
        }
    };

    class Kernel {
      private:
        std::string name;
        IterationSpace *iter;
        std::vector<Accessor*> accs;
        ValueDecl *VD;
        HipaccKernelClass *KC;

      public:
        Kernel(std::string name, IterationSpace *iter,
                ValueDecl *VD, HipaccKernelClass *KC)
            : name(name), iter(iter), VD(VD), KC(KC) {
        }

        std::string getName() {
          return name;
        }

        HipaccKernelClass *getKernelClass() {
          return KC;
        }

        IterationSpace *getIterationSpace() {
          return iter;
        }

        std::vector<Accessor*> getAccessors() {
          return accs;
        }

        ValueDecl *getValueDecl() {
          return VD;
        }

        // TODO move this to image
        std::vector<Accessor*> getAccessors(Image *image) {
          std::vector<Accessor*> ret;
          for (auto it = accs.begin(); it != accs.end(); ++it) {
            if ((*it)->getImage() == image) {
              ret.push_back(*it);
            }
          }
          return ret;
        }

        void addAccessor(Accessor *acc) {
          accs.push_back(acc);
        }
    };

    class Node {
      private:
        bool space;

      public:
        explicit Node(bool isSpace) {
          space = isSpace;
        }

        bool isSpace() {
          return space;
        }
    };

    class Space : public Node {
      friend class Process;

      private:
        Image *image;
        IterationSpace *iter;
        std::vector<Accessor*> accs_;
        Process *srcProcess;
        std::vector<Process*> dstProcess;
        bool isShared;

      public:
        std::string stream;
        std::vector<std::string> cpyStreams;
        explicit Space(Image *image) : Node(true), image(image), iter(nullptr), srcProcess(nullptr), isShared(false) { }
        Image *getImage() {
          return image;
        }

        IterationSpace *getIterationSpace() {
          return iter;
        }

        // only for dump
        std::vector<Accessor*> getAccessors() {
          return accs_;
        }

        Process *getSrcProcess() {
          return srcProcess;
        }

        std::vector<Process*> getDstProcesses() {
          return dstProcess;
        }

        void setSpaceShared() {
          isShared = true;
        }

        bool isSpaceShared() {
          return isShared;
        }

        void setSrcProcess(Process *proc) {
          IterationSpace *iter = proc->getKernel()->getIterationSpace();
          hipacc_require(iter->getImage() == image, "IterationSpace Image mismatch");
          this->iter = iter;
          srcProcess = proc;
        }

        void addDstProcess(Process *proc) {
          // a single process can have multiple accessors to same image
          std::vector<Accessor*> accs = proc->getKernel()->getAccessors(image);
          hipacc_require(accs.size() > 0, "Accessor Image mismatch");
          for (auto it = accs.begin(); it != accs.end(); ++it) {
            accs_.push_back(*it);
          }
          dstProcess.push_back(proc);
        }
    };

    class Process : public Node {
      private:
        Kernel *kernel;
        Space *outSpace;
        std::vector<Space*> inSpaces;
        // todo
        Process* readDependentProcess;
        Process* writeDependentProcess;

      public:
        Process();
        Process(Kernel *kernel, Space *outSpace)
            : Node(false),
              kernel(kernel),
              outSpace(outSpace),
              readDependentProcess(nullptr),
              writeDependentProcess(nullptr)
        {
          outSpace->srcProcess = this;
        }

        Kernel *getKernel() {
          return kernel;
        }

        Space *getOutSpace() {
          return outSpace;
        }

        std::vector<Space*> getInSpaces() {
          return inSpaces;
        }

        void addInputSpace(Space *space) {
          inSpaces.push_back(space);
        }
    };

    template<class T>
    void freeVector(std::vector<T> &vec) {
      for (auto it = vec.begin(); it != vec.end(); ++it) {
        delete (*it);
      }
    }

    template<class T>
    bool findVector(std::vector<T> &vec, T item) {
      return std::find(vec.begin(), vec.end(), item) != vec.end();
    }

    ~HostDataDeps() {
      freeVector(spaces_);
      freeVector(processes_);
    }

    void addImage(ValueDecl *VD, HipaccImage *img);
    void addMask(ValueDecl *VD, HipaccMask *mask);
    void addBoundaryCondition(ValueDecl *BCVD, HipaccBoundaryCondition *BC, ValueDecl *IVD);
    void addKernel(ValueDecl *KVD, ValueDecl *ISVD, std::vector<ValueDecl*> AVDS);
    void addAccessor(ValueDecl *AVD, HipaccAccessor *acc, ValueDecl* IVD);
    void addIterationSpace(ValueDecl *ISVD, HipaccIterationSpace *iter, ValueDecl *IVD);
    void recordVisitedKernelDecl(DeclRefExpr *DRE, std::vector<VarDecl *> &VKD);
    void runKernel(ValueDecl *VD);
    void dump(partitionBlock &PB);
    void dump(edgeWeight &wMap);
    std::vector<Space*> getInputSpaces();
    std::vector<Space*> getOutputSpaces();
    void markProcess(Process *t);
    void markSpace(Space *s);
    void generateSchedule();
    void insertKernelDependencyGraph(Process* srcP, Process* destP);
    void insertDestSpaceDependencyGraph(Process* srcP, Space* destS);
    void insertSrcProcessDependencyGraph(Process* srcP, Space* inS);
    void buildGraphDependency();
    void addMemcpyNodeGraph(std::string imgDst, std::string imgSrc, std::string direction);
    void addKernelNodeGraph(std::string kernelName);
    std::string getMemcpyNodeName(std::string imgDst, std::string imgSrc, std::string direction);
    std::string getKernelNodeName(std::string kernelName);

    // helper to convert a partitionBlock to a block of the respective kernel names
    static partitionBlockNames convertToNames(const partitionBlock* pB) {
      partitionBlockNames PBNam;
      llvm::errs() << " [ ";
      for (auto pL : *pB) {
        llvm::errs() << "{";
        std::list<std::string> lNam;
        for (auto p : *pL) {
          std::string kname = p->getKernel()->getName();
          llvm::errs() << " --> " << kname;
          lNam.push_back(kname);
        }
        llvm::errs() << "} ";
        PBNam.push_back(lNam);
      }
      llvm::errs() << "] \n";

      return PBNam;
    }

    static bool partitionBlockNamesContains(
      const std::set<partitionBlockNames>& haystack,
      const std::string& needle
    ) {
      for (const auto& PBN : haystack) {
        if (std::any_of(PBN.begin(), PBN.end(), [&](std::list<std::string> lNam){
          return (std::find(lNam.begin(), lNam.end(), needle) != lNam.end()) &&
            (lNam.size() > 1);})) {
          return true;
        }
      }
      return false;
    }

    // kernel fusion analysis
    void computeGraphWeight();
    void fusibilityAnalysis();
    void fusibilityAnalysisLinearAndParallel();
    void minCutGlobal(partitionBlock PB, partitionBlock &PBRet0, partitionBlock &PBRet1);
    unsigned minCutPhase(partitionBlock &PB, edgeWeight &curEdgeWeightMap,
        std::pair<Process *, Process *> &ST);
    bool isLegal(const partitionBlock &PB);
  public:
    bool isFusible(HipaccKernel *K);
    bool hasSharedIS(HipaccKernel *K);
    std::string getSharedISName(HipaccKernel *K);
    bool isSrc(Process *P);
    bool isDest(Process *P);
    const std::set<FusiblePartitionBlock>& getFusiblePartitionBlocks() const;
    std::string getGraphMemcpyNodeName(std::string dst, std::string src, std::string dir);
    std::string getGraphKernelNodeName(std::string kernelName);
    std::set<std::string> getGraphMemcpyNodeDepOn(std::string dst, std::string src, std::string dir);
    std::set<std::string> getGraphKernelNodeDepOn(std::string kernelName);
    std::map<std::string, std::set<std::string>> getGraphNodeDepMap() const;
    std::vector<std::string> getOutputImageNames();

    static HostDataDeps *parse(ASTContext &Context,
        PrintingPolicy &Policy,
        AnalysisDeclContext &analysisContext,
        CompilerKnownClasses &compilerClasses,
        CompilerOptions &compilerOptions,
        llvm::DenseMap<RecordDecl *, HipaccKernelClass *> &KernelClassDeclMap) {

      static HostDataDeps dataDeps;
      dataDeps.compilerClasses = compilerClasses;
      dataDeps.compilerOptions = &compilerOptions;
      dataDeps.KernelClassDeclMap = KernelClassDeclMap;
      DependencyTracker DT(Context, Policy, analysisContext, compilerClasses, dataDeps);
      dataDeps.generateSchedule();
      if (dataDeps.compilerOptions->fuseKernels()) {
        dataDeps.fusibilityAnalysisLinearAndParallel();
      }
      if (dataDeps.compilerOptions->useGraph()) {
        dataDeps.buildGraphDependency();
      }

      return &dataDeps;
    }
};

class FusiblePartitionBlock {
  public:
    class KernelInfo;
    using Part = std::vector<KernelInfo>;

    enum class PatternType {
      Linear,
      Parallel
    };

    enum class Pattern {
      // Linear patterns
      Linear,

      // Parallel patterns

      // Parallel points to point
      NP2P,
      // Parallel locals to point
      NL2P,
      // Parallel mixed locals/points to point
      Mixed2P,
      // Parallel points to local
      NP2L,
      // Parallel locals to local
      NL2L,
      // Parallel mixed locals/points to local
      Mixed2L
    };

    struct KernelInfo {
      std::string name;

      const std::string& getName() const;

      bool operator < ( const KernelInfo& rhs ) const;
    };
  
  private:
    Pattern pattern;
    std::vector<Part> parts;
    std::unordered_set<std::string> kernelNames;
    
  public:
    FusiblePartitionBlock(PatternType patternType, HostDataDeps::partitionBlock& inBlock);

    static std::set<FusiblePartitionBlock>::iterator findForKernel(
      const HipaccKernel* kernel,
      const std::set<FusiblePartitionBlock>& fusibleBlocks
    ) {
      return std::find_if(
        fusibleBlocks.begin(),
        fusibleBlocks.end(),
        [&](const FusiblePartitionBlock& block) {
          return block.hasKernel(kernel);
        }
      );
    }

    /**
     * Check whether the pattern of this block is fusible.
     */
    bool isPatternFusible() const {
      // Return true if the pattern is fusible by the current ASTFuse tool, false otherwise.

      switch (pattern) {
        case FusiblePartitionBlock::Pattern::Linear:
        case FusiblePartitionBlock::Pattern::NP2P:
          return true;
        default:
          return false;
      }
    }

    PatternType getPatternType() const;
    Pattern getPattern() const;
    const std::vector<Part>& getParts() const;
    bool hasKernelName(const std::string& name) const;
    bool hasKernel(const HipaccKernel* kernel) const;

    bool operator < ( const FusiblePartitionBlock& rhs ) const;
};

}
}

#endif  // _HOSTDATADEPS_H_

// vim: set ts=2 sw=2 sts=2 et ai:
//
